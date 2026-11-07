# ForestSAR instance review for X-AnyLabeling

This fork adds an optional Qt review dock backed by a separate local DEIMv2/SAM3
worker, plus an RGB/thermal background toggle (`V`). The integration targets
X-AnyLabeling **4.3.1** and 30-fps, person-only sequences. It does not change the
regular `xanylabeling` launch or install new core dependencies. The review dock
uses internal GUI APIs and may need adaptation for other app versions.

## Configure and launch

Copy `forest_sar.example.json` to the ignored `forest_sar.local.json` and set your
local paths. Paths are relative to the JSON file; absolute paths also work.
Set `FOREST_SAR_CONFIG` to the file's absolute path before every command:

```bash
export FOREST_SAR_CONFIG=/path/to/X-AnyLabeling/forest_sar.local.json
python -m annotation_workbench.xanylabeling_addon --sequence GROUP/SEQUENCE --gpu 0
```

Run from this checkout, or install the fork in your chosen X-AnyLabeling runtime.
An installed fork also provides `xanylabeling-review` with the same arguments.
`--output` overrides the review output; `--frame 57` opens a frame; `--port 8793`
selects the local worker port. To reuse another compatible Python installation's
GUI libraries without changing your inference environment, set the optional
`gui_site` path to that installation's site-packages directory. Both installations
must use compatible Python and Qt binaries. CUDA/Qt shared-library paths remain
the responsibility of the local environment.

The review store needs NumPy and Pillow. GUI use needs X-AnyLabeling's normal
PyQt6 dependencies. Inference additionally needs a working PyTorch/CUDA install,
separately installed DEIMv2 and SAM3 dependencies, and locally authorized model
checkpoints. No model weights, videos, annotations, or credentials are distributed.
Use a single-class DEIMv2 config/checkpoint with `person` as class 0; the adapter
rejects other classes. The SAM3 integration uses `Sam3VideoPredictor` request APIs
including `force_tracker_propagation` and `evict_cached_frame_outputs`, tested
against facebookresearch/sam3 commit `f1647e8`. Existing environments can be reused;
this feature adds no mandatory inference packages to the app's dependency list.

## Dataset and cache

```text
DATA_ROOT/
  FRAMES/GROUP/SEQUENCE/SEQUENCE_000001.jpg
  FRAMES/GROUP/SEQUENCE/SEQUENCE_000002.jpg
  ...
  raw_predictions/GROUP/SEQUENCE.npy
  anchor_references.json                 # optional
OUTPUT_ROOT/
  annotations/GROUP/SEQUENCE/SEQUENCE_000001.{jpg,json}
  metadata/GROUP/SEQUENCE/instances.json
  metadata/GROUP/SEQUENCE/segments/SEGMENT_UUID.json
  metadata/GROUP/SEQUENCE/transactions/...
  metadata/GROUP/SEQUENCE/review_queue.csv
```

Each sequence must contain contiguous, one-based JPEGs with constant dimensions.
`raw_predictions` is configurable and contains one NumPy array per sequence,
shaped `(frames, queries, 5)`: pixel-space `x0,y0,x1,y1,score`. Retain low-score
candidates in this cache. It must come from the same detector, frames, and image
geometry used for anchor refresh. Preparing this dense cache is a prerequisite;
this add-on refreshes anchors but does not produce the complete dense cache.

Optional sparse references use the following manifest. Keys are one-based dense
frame numbers, and bounding boxes are pixel XYXY coordinates. An empty list means
an explicitly sampled empty anchor; an absent key means an unlabeled frame.

```json
{
  "GROUP/SEQUENCE": {
    "split": "train",
    "frames": {"1": [{"bbox": [10, 20, 40, 80]}], "31": []}
  }
}
```

Set `anchor_references` to another manifest path if needed. The original research
alignment format is also supported by configuring `anchor_alignment` (an
`anchors` array with group, base_sequence, split, coco_image_id,
dense_frame_1based and status=`exact_pixel_match`) and `coco_annotations` (a
directory containing split COCO JSON files). A manifest takes precedence. With no
references, generation uses frames 1, 31, 61, ... as detector seeds and reruns DEIM
on each of them. Reference frames need not share that exact phase.

Initialization creates editable Labelme-style JSON files and relative image
symlinks in the output directory. Keep metadata with the labels: it holds the
instance lineage and reversible operations. Source images and sparse references
are inputs; review edits go to the output directory.

## Review workflow

Use the dock as follows:

1. **Open source sequence…** chooses a source sequence and initializes its editable
   review directory. **Generate labels for this sequence** runs the pipeline below.
   Generation can resume: already completed seed proposals are remembered.
2. Draw a rectangle with X-AnyLabeling's rectangle tool, label it `person`, and
   select it. Click **Track selected box → 1 second**. This works on sparse anchors
   and intermediate frames. It follows up to 30 transitions, including the seed
   image and at most 30 following images. The sequence boundary limits the interval.
3. Selecting an existing linked box extends its instance. A newly drawn box
   receives an instance UUID and numeric group ID, or joins a clearly overlapping
   existing segment. Double-click an instance row to jump to its first frame.
4. Select one linked box or an instance row and click **Delete entire selected
   instance** to remove every label with that instance UUID, including related
   segments and imported human anchor shapes in the review copy. Other instances
   remain. Rejected observations are remembered during automatic regeneration.
5. **Undo last worker operation** restores the operation's before-images. Undo
   refuses to overwrite newer manual edits. **Accept entire selected instance**
   explicitly marks all its related labels reviewed; inspect the track's extent
   before accepting it. Automatic outputs remain pending until this action.
6. Correct individual boxes normally in X-AnyLabeling and save. The add-on marks
   geometry changes as manual so automatic regeneration preserves them. Existing
   accepted boxes are also preserved. An explicit new tracking request revises
   the requested interval and marks new predictions pending.

The instance list can be filtered to pending review, disagreements with human
anchors, detector-only proposals, or forward tracks without backward confirmation.
`DEIM` and `Back` count frames supported by geometric detector agreement and
backward agreement respectively. `Person` records whether any seed was also
matched by SAM3's separate text-only `person` detector. None of these automatically
accepts an instance. Select a row to inspect its seed and anchor disagreements.
Generation, tracking, deletion and acceptance save current edits first. Worker
operations temporarily disable editing and reload the current frame when
finished. Undo requires current manual edits to be settled first. Progress and
errors appear in the dock.

Automatic instance admission now requires both a matched DEIM score **strictly
above 0.5** somewhere in its observations and a run of **at least 10 consecutive
observed frames**. Observations may be DEIM boxes or nonempty forward SAM3 masks.
The score is matched to the object's box using IoU >=0.3, never borrowed from an
unrelated high-score detection elsewhere in the image. A score of exactly 0.5
does not pass. A span of 10 frames with sparse gaps does not by itself pass.

After admission, later low DEIM scores or missing DEIM detections do not remove
the continuing SAM3 masks. An extension inherits its family's earlier strong
evidence. Automatic seed tracking is probed before allocating a new instance;
failed probes are recorded without adding labels. Detector-only tracklets also
pass the instance policy before generation finishes. **Filter automatic
instances** applies these rules to existing proposals as one undoable operation.

Before filtering, a short fragment may join a unique continuation across at most
one missing frame, with boundary IoU >=0.5 and an unambiguous best match. At least
one side must already meet admission. This prevents a six-frame continuation of
an established person from being mistaken for a new short-lived instance. UUID
aliases and segment provenance preserve the association for deletion and undo.
Longer occlusions retain separate segments for review. No fixed limit on the
number of people is imposed by the filter.

Explicitly accepted instances, manual seeds and manually corrected geometry are
protected from automatic pruning. Original unreviewed sparse annotations still
undergo the automatic policy. Automatic filtering does not create human-rejection
tombstones, and it does not overwrite the earlier review revision.

The revised automatic pipeline uses frozen models:

- Rerun DEIMv2-X at 640×640 on every 1-fps anchor, including anchors with no human
  boxes, plus frame 1. Reuse the original dense 300-candidate cache elsewhere.
- Consider candidates with score **≥0.1**, NMS IoU 0.6, and containment suppression
  at 85% coverage of the smaller box. Human reference boxes and unmatched DEIM
  candidates at anchor frames initialize one-second tracks. Remaining dense
  DEIM candidates appear as detector-only proposals, also on anchor images.
- Initialize SAM3 using an instance box prompt (corner labels 2 and 3). This
  avoids requiring SAM3's concept detector to first recognize a small thermal
  person. Retain **every nonempty forward mask**, using its bounding box directly.
  The stricter old Kalman-track export filter and bidirectional consensus gate
  are not applied to these proposals.
- If an independent endpoint human/DEIM box exists, run backward tracking and
  record agreement at IoU ≥0.3. Missing or failed backward verification never
  removes a forward candidate.
- Separately run SAM3 with the text `person` and no box prompt on seed/anchor
  images. Record its agreement as independent semantic evidence. Geometric
  agreement with a box-conditioned SAM3 mask alone does not establish that a
  candidate is a person. Failure of the text detector also does not invalidate
  a visible person or a manual seed.
- Treat original human boxes as editable reference proposals, rather than a
  completeness constraint. Keep their immutable reference coordinates in each
  frame's `human_reference` field. Additional proposals are allowed at empty or
  incomplete anchors; mark disagreements for human review.

Every generated shape has `sar.instance_id`, `sar.shape_id`, origin, and review
state. Propagated shapes also carry their segment ID, seed frame/box, and
verification evidence. Each segment retains its complete forward/backward box
trace. Multiple segments can share one instance UUID. Association uses overlap
and remains fallible; IDs are sequence-local annotation identities, not a claim
of guaranteed person re-identification through long occlusions.

Automatic proposals need human review. Consistent SAM3 tracking of a wrongly
seeded background region can still produce false positives. Association uses
geometric overlap and can split or join people incorrectly; long occlusions do
not guarantee identity continuity.

## Headless commands

```bash
python -m annotation_workbench.cli initialize GROUP/SEQUENCE
python -m annotation_workbench.cli generate GROUP/SEQUENCE --gpu 0
python -m annotation_workbench.cli track GROUP/SEQUENCE --frame 57 --box 10 20 40 80 --gpu 0
python -m annotation_workbench.cli filter GROUP/SEQUENCE
python -m annotation_workbench.cli delete GROUP/SEQUENCE --instance-id INSTANCE_UUID
python -m annotation_workbench.cli undo GROUP/SEQUENCE
```

Use `--cached-anchors` to reuse stored anchor predictions and `--no-backward` to
skip backward verification. Do not mutate a sequence headlessly while it is open
for editing. The GUI saves and reloads around its own worker operations. The
worker binds to 127.0.0.1 with a private token stored in `metadata/backend.json`.
Logs are in `metadata/backend.log` and `metadata/jobs/`. Closing the GUI leaves the
worker alive; stop its recorded PID after jobs finish to release GPU memory.

## RGB/thermal background toggle

The `V` shortcut and **Toggle RGB/Thermal** action look for paired paths under
`images` and `images_ir`, optionally replacing filename suffix `_rgb` with `_ir`.
The pair must share image dimensions and already be aligned. Shapes, the label
filename, and original embedded image bytes stay attached to the opened source
image. This switch changes the display; it does not register the modalities or
switch the inference input.

## Validation

```bash
python -m unittest annotation_workbench.test_workbench annotation_workbench.test_quality annotation_workbench.test_settings -v
QT_QPA_PLATFORM=offscreen FOREST_SAR_RUN_GUI_TESTS=1 python -m unittest annotation_workbench.test_gui -v
```

The first command uses generated images and fake tracker outputs, needs no private
data/checkpoints, and tests lineage, deletion, undo, edit preservation, admission
and configuration. The optional GUI checks use actual app widgets and synthetic
images; they do not measure detector or tracker accuracy. GPU inference requires
your own frames, caches, model source trees, and checkpoints. Unit tests should be
run with `FOREST_SAR_CONFIG` unset, so they cannot select production caches.

The integration is distributed under the repository's GPLv3 license. SAM3 and
DEIMv2 remain separately installed projects with their own terms.
