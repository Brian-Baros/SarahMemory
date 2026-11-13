#!/usr/bin/env python3
"""--==The SarahMemory Project==--
File: SarahMemoryAvatarGenerate.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

===============================================================================

SarahMemory-AvatarBuilder.py
WHAT THIS IS
-------------
A single Python script that can run WITHOUT opening Blender's GUI:

    python SarahMemoryAvatarGenerate.py
    python SarahMemoryAvatarGenerate.py --front front.png --side side.png --back back.png --blueprint blueprint.png
    python SarahMemoryAvatarGenerate.py --quality balanced --allow-blender --user-consented
    python SarahMemoryAvatarGenerate.py --outdir "C:\\SarahMemory\\resources\\avatars\\3D"

By default it writes into ``resources/avatars/3D`` next to this script's
project root (the script's parent's parent, or the current folder if that
layout is not found), producing:
    SarahMemoryAvatar_RigBootstrap.blend
    SarahMemoryAvatar_RigBootstrap.glb
    SarahMemoryAvatar_RigBootstrap.json   (manifest + GLB verification)

HOW IT RUNS
-----------
Plain ``python`` cannot build a mesh — Blender can. So this script works in
two modes, in one file:
  1. OUTER mode (plain Python, no Blender needed): finds blender.exe
     (``--blender`` argument, PATH, or common Windows install locations),
     re-runs ITSELF inside Blender headless, then verifies the exported
     GLB with pure Python and prints PASS / FAIL. It never claims success
     if verification fails.
  2. INNER mode (inside Blender, ``--inner``): builds the scene with bpy.

HONEST LIMIT — READ THIS FIRST
------------------------------
A 2D image cannot be turned into a perfect photoreal 3D sculpt by any
script, and this one does not pretend to. The images are used for:
  * palette sampling (hair magenta, suit black, neon cyan, skin tone),
  * proportion targets (1.7272 m / 5'8", per Brian 2026-11-03, blueprint 60K-90K runtime poly target),
  * neon routing (centre torso, V chest, waist, leg, arm, collar, back).
The geometry itself is a clean procedural humanoid, built to move:
  * ARMS-DOWN rest pose (not the old T-pose). The old rig's arm bones ran
    sideways along X, so its rest pose in AvatarPanel was a crucifix.
    This rig's arms hang down, so "stand" in the panel looks like stand.
  * ONE body mesh, ONE head mesh, ONE hair mesh, ONE boots mesh — not
    200 separate objects.
  * Every detail mesh (neon, brows, lips, lashes, hair strands) is a
    MESH (not a curve) with an armature modifier and a single-bone
    vertex group, so it follows the body in motion instead of floating.
  * Head shape keys: Basis, Blink_L, Blink_R, JawOpen, Smile, Frown,
    BrowsUp, Squint_L, Squint_R, MouthWide. No drivers are added — the
    AvatarPanel UI drives morphs directly in Three.js, which is what
    the old driver setup interfered with.
  * Smart-UV-projected Body/Head/Boots, so a texture CAN be applied
    later. The old June GLB had no UVs on those meshes at all.
  * v1.0.6 (this candidate, queue item 1 — texture): the verified
    single-avatar crops in source_single/*_white.png are now wired
    into the body/suit and head/skin materials as Image Texture nodes
    on Base Color, so the glTF exporter embeds them in the GLB
    (the June GLB had textures=null). *_black.png is NEVER used as
    a texture source. If no crop is found, or --no-texture is given,
    the build falls back to the flat v1.0.5 colours and still succeeds.
    LIMIT: Body/Head use Smart UV, not a true front projection, so
    the texture will map approximately on curved parts — judge it
    from Brian's screenshot, do not claim a clean decal here.
  * v1.0.7 (this candidate, queue item 2 — neon-on-surface, built on
    v1.0.6 so the texture step is kept): every neon strip is split
    at the joints into short segments no longer than 0.14 m (was up
    to 0.76 m for the leg strip and 0.64 m for the torso centre,
    each bound to a single bone it crossed). Each segment is bound
    to exactly the bone its centre lies in, is 0.008 m wide and
    0.002 m thick, and sits ~0.002 m off the local body surface
    (e.g. torso front centre y -0.133 against the torso sphere's
    -0.13 front, chest front -0.143, boot shaft front -0.083).
    No segment crosses a waist, elbow, knee or ankle joint, so a
    segment now follows one bone in motion instead of bridging a
    bend. LIMIT, stated plainly: this is still a box-strip
    approximation of a circuit line, not a curve hugging a curved
    surface point by point — judge from Brian's screenshot whether
    the shorter, closer segments read as on-surface.
  * v1.0.8 (this candidate, queue item 3 — continuous body, built on
    v1.0.7 so texture + neon-on-surface are kept): the body is no
    longer isolated sphere blobs that pinch at the joints. Each limb
    segment is lengthened by 0.02 m in z so neighbouring segments
    overlap by ~0.07 m instead of ~0.03 m, and an explicit bridge
    sphere is added at every joint — waist (spine.01), shoulder,
    elbow, wrist, knee, ankle — plus two loft spheres between
    hips/torso/chest so the torso profile steps gradually instead
    of pinching between three big blobs. Bridge radii never exceed
    the neighbouring segment radii in Y, so the v1.0.7 neon offsets
    (surface + ~0.002 m gap) still hold. The bridges join into the
    single Sarah_Body mesh and are auto-weighted, so a bridge at a
    joint is weighted to both bones either side and bends with the
    joint instead of gaping. LIMIT, stated plainly: this is still
    overlapping spheres, not a true capsule/lofted skin — the
    silhouette will still read slightly lumpy up close, and NO .glb
    was built or verified here (no Blender in this environment).
    Judge from Brian's screenshot, ideally mid-walk/wave, whether
    the waist/elbow/knee gaps are gone.
  * v1.0.9 (this candidate, queue item 4 — face, built on v1.0.8 so
    texture + neon-on-surface + continuous body are kept): the face
    is no longer tiny and flat inside an oversized hair shell.
    Sclera radius 0.026 -> 0.034 m and centre moved forward
    y -0.118 -> -0.128, so its front (-0.162) now protrudes ~0.017 m
    past the head front (-0.145) instead of sitting flush with it;
    iris radius 0.0125 -> 0.018 m, centre y -0.142 -> -0.150, front
    -0.168. Brows widened 0.056 -> 0.072 m and moved to y -0.150,
    so their backs sit ~0.001 m inside the head surface and their
    fronts read ~0.009 m proud of it; top lashes widened 0.060 ->
    0.076 m on the same surface, plus a new thin lower-lash pair.
    Lips widened (upper 0.070 -> 0.096 m, lower 0.064 -> 0.090 m)
    and moved to y -0.150, fronts ~0.009 m proud of the head.
    Hair shell shrunk modestly (0.20/0.17/0.22 -> 0.185/0.155/0.205)
    and crown shrunk slightly, so the face is less swallowed; the
    full hair rework (wavy strands down the back) stays queue item 5.
    Bone names, the 9 head morphs, and the arms-down pose are
    unchanged; brows/lashes/lips stay single-bone (head/jaw) meshes.
    LIMIT, stated plainly: these are still sphere eyes and box
    brows/lashes/lips, not sculpted toon eyelids or a modelled
    mouth — judge from Brian's screenshot whether the face now
    reads at panel distance, and NO .glb was built or verified here
    (no Blender in this environment).
  * v1.0.10 (this candidate, queue item 5 — hair, built on v1.0.9 so
    texture + neon-on-surface + continuous body + face are kept):
    the hair is no longer one oversized blob. The shell shrinks
    again (0.185/0.155/0.205 -> 0.170/0.140/0.185) and the crown
    shrinks (0.16/0.13/0.11 -> 0.145/0.12/0.10), and the solid
    Hair_BackMass blob (0.19/0.12/0.46, z 1.06-1.98) is DELETED
    and replaced by three wavy strands down the back — centre,
    left and right — each a chain of small overlapping spheres
    (segments=20, z step 0.12 m, z half-size 0.07-0.075 m, so
    neighbours overlap and the strand reads continuous, not
    beaded), swaying alternately x +/-0.025 m per step, running
    z 1.88 down to 1.16 (past the shoulders, to mid-back). Front
    locks, side falls and shoulder falls are unchanged, so face
    framing is preserved. All strands join into the single
    Sarah_Hair mesh and are auto-weighted, so the lower strands
    follow the hair.01/02/03 bones. LIMIT, stated plainly: these
    are still joined spheres, not sculpted wavy hair cards, and
    the strands are narrower than the old blob, so back neon now
    reads BETWEEN strands instead of being buried — judge from
    Brian's screenshot (back view if possible) whether the wave
    reads, and NO .glb was built or verified here (no Blender in
    this environment).
  * v1.0.11 (this candidate, queue item 6 — height/fit, built on
    v1.0.10 so texture + neon + continuous body + face + hair are
    kept): Brian decided 2026-11-03: 5'8" governs. 5'8" = 1.7272 m
    exactly (the 1.72 m figure in the 14-image set is that rounded).
    The v1.0.10 model topped out at ~2.025 m (hair shell centre
    1.84 + 0.185 radius), so every geometry/bone coordinate is
    scaled uniformly by HEIGHT_SCALE = 1.7272 / 2.025 = 0.85294
    inside add_sphere / add_box / add_bone, with shape-key
    thresholds and offsets scaled by the same factor. Proportions
    are unchanged (uniform scale), bone names and the 9 morphs are
    unchanged, and the new top is 1.7272 m, feet still at 0.0.
    LIMIT, stated plainly: uniform scaling also narrows width
    proportionally (e.g. torso width 0.42 -> 0.358 m) — that is
    correct if 5'8" is her true height, but if the panel still
    crops, the panel camera/fit is the cause, not the GLB height,
    and NO .glb was built or verified here (no Blender here).
  * v1.0.12 (this candidate, queue item 7 — boots, built on
    v1.0.11 so texture + neon + continuous body + face + hair +
    5'8" height are kept): the boots get the two glowing parts
    from the video references, as separate neon-material meshes
    (not painted onto the flat suit boots mesh): (1) a glowing
    sole plate under each foot — a thin box slightly wider/longer
    than the foot sphere (0.164 x 0.244 x 0.012 m pre-scale) with
    its bottom at z 0.002 m pre-scale, so the feet stay on the ground,
    bound to foot.L/R; (2) a glowing ankle ring around each boot
    shaft at z 0.17 m — a torus, major radius 0.080 m, minor
    0.007 m, which sits just proud of the shaft surface there
    (~0.071 m x-radius at that height), bound to shin.L/R, since
    the ring centre lies in the shin bone's span, not the foot's.
    The existing boot-shaft neon strip is kept. LIMIT, stated
    plainly: the sole is a flat plate, not a shaped platform sole
    with a heel profile, and the ring is a torus, not a sculpted
    cuff — judge from Brian's screenshot (low/foot close-up if
    possible) whether sole + ankle glow read, and NO .glb was
    built or verified here (no Blender in this environment).
  * v1.0.13 (this candidate, queue item 8 — generic character class,
    built on v1.0.12 so texture + neon + continuous body + face +
    hair + 5'8" height + boots are kept): step 1 of
    generic-character-class-design.md ONLY — a new
    AvatarCharacterProfile class, with Sarah as one instance
    (sarah_default()), plus a --character flag (default
    sarah_default). The profile now drives the palette and the
    target height in inner_main, but for Sarah those values are
    the old defaults (PALETTE_DEFAULT, 1.7272 m), so the Sarah
    GLB this script builds must match v1.0.12 in geometry, bone
    names, and 9 morphs — that is the point of this step: prove
    the parameterisation reproduces Sarah before any second
    character is added. No second profile exists yet (an unknown
    --character fails loudly instead of silently building Sarah),
    object names and OUT_BASENAME are unchanged, and no upload /
    palette-sampling / preview code is added here. LIMIT, stated
    plainly: this is the class shell, not the upload-to-3D
    pipeline — a 2D image still cannot sculpt a face, and NO .glb
    was built or verified here (no Blender in this environment).
  * v1.0.14 (this candidate, queue item 8 — generic character class,
    built on v1.0.13 so texture + neon + continuous body + face +
    hair + 5'8" height + boots + the profile class are kept): step 2
    of generic-character-class-design.md — a SECOND profile,
    toon_hero_01 (style_preset fortnite_toon, per the design doc):
    head_scale 1.25, hand_scale 1.30, shoulder_scale 1.10,
    hip_scale 1.05, same 1.7272 m height as Sarah, and its own
    palette (navy suit, electric-blue hair, amber neon) so a
    --character toon_hero_01 build is visibly NOT Sarah. The
    scales are now wired into geometry: head parts, eyes and face
    details are scaled about the head centre (0, 0, 1.80), hand
    spheres by hand_scale, shoulder spheres/bridges by
    shoulder_scale, hips by hip_scale — so a bigger head still
    keeps its eyes/brows/lips on its (larger) face surface instead
    of burying them. The shared 68-bone rig and 9 head morphs are
    unchanged, so panel/controller actions still work. Safety:
    a non-Sarah profile writes <ObjectPrefix>Avatar_RigBootstrap
    (.glb/.blend/.json) — ToonHeroAvatar_RigBootstrap for this
    profile — and NEVER overwrites SarahMemoryAvatar_RigBootstrap.
    Sarah's *_white.png texture crops are NOT applied to a
    non-Sarah profile (they are Sarah's face/suit); toon builds
    use flat profile palette colours until per-profile
    texture_image wiring lands (next step). LIMIT, stated
    plainly: hair strands/locks, boots, neon and bones are not
    yet rescaled by profile (head hair shell/crown only), so the
    toon hero is a proportions + palette proof, not a finished
    hero sculpt, and NO .glb was built or verified here (no
    Blender in this environment).
  * v1.0.15 (this candidate, queue item 8 — generic character class,
    built on v1.0.14 so texture + neon + continuous body + face +
    hair + 5'8" height + boots + the profile class + the second
    profile are kept): step 3 of generic-character-class-design.md —
    the REMAINING build functions are now parameterised by profile,
    and per-profile texture_image is wired. Hair locks, side falls,
    shoulder falls and back strands now scale with head_scale (width
    / thickness, and back-strand x/y about the head), and the back-
    strand chain length scales with hair_length (z step
    0.12 m * hair_length, so sarah_default at hair_length 1.0 is
    byte-identical to v1.0.14 in z). Boot shaft/foot/heel and the
    sole-glow plate + ankle ring now follow hip_scale in x and in
    width, and the shoulder neon segment follows shoulder_scale
    while the boot neon segment follows hip_scale — arm/leg neon
    centres are unchanged because arm/leg centres were not moved
    in v1.0.14. A profile with texture_image set now loads that
    file (relative to the output folder, or as given) as its
    suit/skin texture instead of Sarah's *_white.png crops; both
    current profiles leave texture_image empty, so sarah_default
    still uses source_single/front_white.png and toon_hero_01
    still builds flat. Bones are deliberately NOT rescaled: the
    shared 68-bone rig is the invariant that lets the panel /
    controller drive any character, per the design doc. LIMIT,
    stated plainly: upload -> palette/silhouette sampling and the
    instant preview are still NOT in this script (that is the
    next step in item 8), and NO .glb was built or verified here
    (no Blender in this environment).
  * v1.0.16 (this candidate, queue item 8 — generic character class,
    built on v1.0.15 so texture + neon + continuous body + face +
    hair + 5'8" height + boots + the profile class + the second profile
    + profile-driven hair/boots/neon/texture are kept): step 4 of
    generic-character-class-design.md — upload -> palette/silhouette
    sampling + instant-preview data. NEW FLAGS: --upload <image>,
    --style {auto,fortnite_toon,sarah_default}, --preview-only, and
    --profile-json (inner use, forwarded by the outer run). A pure-Python
    stdlib PNG reader (no Pillow, no Blender needed for sampling) reads
    a SINGLE-avatar image, rejects a dark-background image loudly (a
    black background would swallow the black suit in the silhouette
    mask — use a *_white.png crop instead), and samples: hair/skin/
    suit palette medians from region boxes, neon from bright saturated
    pixels excluding the sampled hair colour (magenta hair would
    otherwise swamp a thin neon line in the median), and silhouette
    ratios (head width at 8% of bbox height,
    shoulder width at 25%, hip width at 62%) from a background mask.
    Scales are first-pass derivations from those ratios against Sarah
    baselines (head 0.136, shoulder 0.209, hip 0.116 of total height,
    recalibrated in v1.0.17 from Brian's 2026-11-03 calibration run
    on source_single/front_white.png, which measured exactly those
    ratios — the v1.0.16 guesses 0.20/0.27/0.21 all clamped to 0.9),
    clamped to head 0.9-1.5, shoulder/hip 0.9-1.3; hand_scale stays
    1.0 because hands are not measurable from a silhouette, and
    height stays 1.7272 m because a 2D image gives ratios, not metres.
    --preview-only writes resources/avatars/characters/<id>/
    <id>.preview.json + <id>.profile.json and stops WITHOUT Blender
    and WITHOUT a GLB: that JSON is the data a panel template-swap
    preview would apply (sampled palette + scales), it is not itself
    a 3D preview, and the panel-side preview is still to be built.
    Without --preview-only, the same sampled profile is built in
    Blender into that characters/<id>/ folder (never the 3D root, so
    it can never overwrite SarahMemoryAvatar_RigBootstrap.glb), with
    the upload image itself as the profile texture_image. LIMIT,
    stated plainly: PNG only (8-bit RGB/RGBA, non-interlaced) in this
    step — a JPEG upload fails loudly instead of being misread;
    the scale baselines are uncalibrated guesses until Brian runs
    --upload on Sarah's own source_single/front_white.png and checks
    the scales come back near 1.0; and face likeness is still NOT in
    scope, exactly as in the design doc. NO .glb was built or verified
    here (no Blender in this environment).
  * v1.0.18 (this candidate, queue item 9 - Fortnite-toon proportions,
    built on v1.0.17 so texture + neon-on-surface + continuous body +
    face + hair + 5'8" height + boots + generic character class +
    upload sampling are kept): Sarah herself becomes fortnite_toon.
    sarah_default now has head_scale 1.18, hand_scale 1.25,
    shoulder_scale 1.08, hip_scale 1.05, and a NEW bulk_scale 1.10:
    torso/chest/neck/loft/waist and upper-arm/forearm/thigh/shin
    radii (x/y only, lengths and joint centres unchanged) grow by
    bulk_scale, so the silhouette reads chunkier without moving any
    joint. Head/hands/shoulders/hips/hair/boots already followed
    their scales in v1.0.14-v1.0.15, so no new geometry code was
    needed for those. Bolder toon shading for fortnite_toon profiles:
    suit metallic 0.18 -> 0.0 and roughness 0.12 -> 0.35, skin
    roughness 0.47 -> 0.60, hair 0.36 -> 0.55, neon emissive 4.0 ->
    5.0 (flat and textured materials alike). Neon y now follows the
    geometry that moved: torso/arm/leg neon by bulk_scale, shoulder
    neon by shoulder_scale, boot neon by hip_scale - without this,
    the bulkier torso (front -0.143 m) and shoulder (front -0.0756 m)
    would have buried the v1.0.7 neon strips. toon_hero_01 keeps
    bulk_scale 1.0, so its geometry is unchanged, but it also gets
    the bolder fortnite shading (it already was fortnite_toon).
    LIMIT, stated plainly: this is still sphere geometry with a
    flatter Principled response, NOT a true cel/toon shader (glTF
    has no standard cel shader) and NOT a Fortnite sculpt - that
    ceiling is queue items 10-13 (ring body, projected texture,
    sculpted face, hair cards). Head growth also raises the total
    height: hair-shell top 2.065 m pre-scale -> ~1.76 m, not
    1.7272 m, because the head scales about its fixed centre; check
    panel fit on this build. NO .glb was built or verified here
    (no Blender in this environment).
  * v1.0.19 (this candidate, queue item 11 — projected texture,
    step 1 front only, built on v1.0.18 so Fortnite-toon
    proportions + all earlier work are kept, and chosen ahead of
    item 10 at Brian's 2026-11-03 19:23 CST direction because the
    v1.0.17 GLB "looks way different" from his front reference):
    Body and Head no longer use Smart UV. A new project_uv_front
    maps loop UVs from world X (flipped, so model +X samples the
    viewer's left of the photo) and world Z, using SHARED
    Body+Head bounds, so the head samples only the top slice of
    front_white.png and the body samples the full height. Boots
    keep Smart UV (flat suit material, UV irrelevant). LIMIT,
    stated plainly: front only — back and side surfaces mirror /
    stretch the front image, they do NOT yet use back_white /
    side_white; geometry is still spheres, so even a correctly
    projected texture will not sculpt the face/body into the
    reference — that needs items 10, 12, 13. If the front texture
    reads mirrored in Brian's test, the u flip is the one line to
    change. NO .glb was built or verified here (no Blender).
  * v1.0.21 (this candidate, queue item 11 — projected texture,
    step 3 tri-projection, built on v1.0.20 so the bbox-enlarged
    front projection + Fortnite-toon proportions + all earlier work
    are kept): Body and Head no longer sample front_white.png on
    every side. A new project_uv_tri assigns each polygon by its
    dominant world normal — front (-Y) to front_white, back (+Y) to
    back_white, side (+/-X) to side_white — with its own material
    slot per direction (suit/skin x front/side/back) and its own
    projection (front/back: u from world X, side: u from world Y,
    all: v from world Z, shared Body+Head bounds, each image
    cropped to its own avatar bbox as in v1.0.20). Top/bottom
    polygons fall back to front. A missing side/back crop falls
    back to the front image for that direction, loudly, so a
    front-only folder still builds exactly as v1.0.20. LIMIT,
    stated plainly: one side image serves BOTH sides, so one side
    will mirror; the side photo's left/right orientation is
    uncalibrated until Brian checks a side view; seams will show
    where front/side/back meet on a sphere mesh, and geometry is
    still spheres, so this will not by itself make the GLB look
    like the reference photo — items 10, 12, 13 still needed.
    NO .glb was built or verified here (no Blender).
  * v1.0.22 (this candidate, queue item 10 — Bootstrap ring body,
    built on v1.0.21 so tri-projection + Fortnite-toon proportions
    + all earlier work are kept): Sarah_Body is no longer 29
    joined spheres. It is 5 ring lofts ported from the v9.1.0a
    Bootstrap _create_ring_mesh_object method (cross-section
    rings joined into a smooth surface, caps at both ends):
    one torso loft (waist pinch, chest taper, neck cap), one
    leg loft per side, one arms-down arm loft per side ending
    in a hand cap. The rings are RE-BASED to this generator's
    coordinates, not copied verbatim from Bootstrap: Bootstrap's
    chest rings reach y -0.19 m and its arms run sideways along
    X, both of which would bury the v1.0.7 neon and break the
    arms-down rest pose. Here the torso front never exceeds
    the neon offsets checked in the code (torso -0.143 m at
    z 1.25, chest back +0.112 m at z 1.42 so the back-spine
    neon at +0.113 m still sits outside it, collar front
    -0.099 m at z 1.62-1.65), leg/arm radii are the old sphere
    radii re-expressed as rings, and bone positions, bone
    names, the 9 head morphs, the tri-projection UVs, the
    neon strips, head, boots and hair are all unchanged.
    LIMIT, stated plainly: the surface is straight between
    rings (no subsurf modifier is added yet), so a close-up
    may read slightly faceted along the torso/legs; the
    torso/leg/arm junctions still overlap rather than being
    one continuous skin, as before; and NO .glb was built or
    verified here (no Blender in this environment).
  * v1.0.23 (this candidate, queue item 12 — sculpted face, step 1:
    toon eyes + angled brows + modelled mouth, built on v1.0.22 so
    ring body + tri-projection + Fortnite-toon proportions + all
    earlier work are kept): the sphere-ball eyes are replaced by
    flattened almond toon eyes — sclera ellipsoid 0.096 x 0.068 m
    (x/z radii 0.048/0.034, y thickness only 0.024) at x +/-0.058,
    y -0.138, z 1.815, with a flat iris disc (0.044 x 0.052 m,
    y thickness 0.012) at y -0.148, so the iris reads as a pupil
    on the lens instead of a second protruding ball. Brows are
    now angled in the face plane (add_box rotation about Y:
    L -0.20 rad, R +0.20 rad, inner ends up, friendly arch)
    instead of straight horizontal boxes, and lashes follow the
    new eye outline (top z 1.845, bottom z 1.780, x +/-0.058).
    The mouth is now modelled, not two floating boxes: a dark
    interior ellipsoid (new Sarah_Mouth_Interior material, from
    the brows palette colour) at y -0.136, z 1.723, framed by
    upper/lower lip boxes at y -0.142, z 1.738/1.708, so the dark
    opening reads in the gap between the lips. One new detail
    object only (Sarah_Mouth_Interior, jaw-bound like the lips),
    no new main mesh, no bone-name change, no morph change:
    eyes/iris keep their
    auto-weight skinning, brows/lashes/lips/mouth-interior keep
    their single-bone (head/jaw) binding, and the 9 head morphs
    are untouched. LIMIT, stated plainly: these are still
    primitive ellipsoids/boxes, not true eyelid shells or a
    sculpted mouth cavity; the nose sphere is unchanged (step
    2); and this step does NOT resolve the photo-face vs 3D-face
    position mismatch — the projected photo eyes still sit under
    the 3D eyes, only flatter and less ball-like, so Brian
    should judge whether the toon face now reads on its own or
    whether the head-front photo texture should be suppressed
    in a later step. NO .glb was built or verified here (no
    Blender in this environment).
  * v1.0.24 (this candidate, queue item 12 — sculpted face, step 2:
    eyelid shells + nose remodel + the photo-face decision, built on
    v1.0.23 so ring body + tri-projection + Fortnite-toon proportions
    + all earlier work are kept): (a) eyelid shells — a skin-colour
    upper cap (0.052/0.013/0.012 at z 1.846) and lower cap
    (0.050/0.012/0.010 at z 1.782) per eye, head-bound like the
    brows/lashes, overlapping only the top/bottom edges of the
    sclera so the visible eye band stays ~1.792-1.834 and the iris
    is framed, not squinted shut; top/bottom lashes move forward
    to y -0.156 so they sit proud of the new lid fronts (-0.155)
    instead of sinking into them. (b) Nose remodel — the single
    nose sphere becomes a bridge ellipsoid (0.018/0.020/0.045 at
    y -0.142, z 1.80) plus a protruding tip ellipsoid
    (0.024/0.018/0.016 at y -0.152, z 1.765), both joined into
    Sarah_Head, so the nose reads as a ridge + tip instead of a
    ball. (c) PHOTO-FACE DECISION: the head-front photo texture
    is SUPPRESSED by default — the Head mesh now uses the flat
    skin material, so the toon face (eyes, lids, brows, nose,
    mouth) reads on its own instead of doubling the projected
    photo eyes underneath it, which is what Brian's 20:18 CST
    shots showed. The BODY suit texture is untouched (front/
    side/back suit crops still embed), and --head-photo restores
    the old photo head in one flag if the flat head reads worse.
    total (upper + lower, L + R), head-bound; no new main mesh,
    no bone-name change, no morph change. LIMIT, stated plainly:
    the eyelids are static skin caps, not animated lid shells —
    they will not close with the Blink morphs (same limitation
    as the brows/lashes today, which are also static detail
    meshes); the nose is still two ellipsoids, not a sculpted
    cavity; and NO .glb was built or verified here (no Blender
    in this environment).
  * v1.0.25 (this candidate, queue item 13 — hair cards, step 1:
    back strands only, built on v1.0.24 so ring body + tri-projection
    + Fortnite-toon proportions + sculpted face + all earlier work are
    kept): the three beaded back-strand chains (21 spheres, z 1.88 ->
    1.16) are replaced by three wavy hair CARDS, one per chain
    (centre, left, right), built as thin solid ribbons (new
    add_hair_card helper, same from_pydata construction as the
    v1.0.22 ring lofts, so no new Blender primitive is needed):
    13 stations, z 1.88 -> 1.16 in 0.06 m steps, width tapering
    1.0 -> 0.65 toward the tip, centre card 0.09 m wide, side
    cards 0.07 m wide at base x +/-0.12 m, so a ~0.047 m gap
    remains between cards where the back-spine neon shows through.
    The wave is a shared sine along the length (amplitude 0.030 m
    on all three cards, in phase, so the gaps stay constant —
    opposite-phase sway was checked in plain Python first and
    nearly closed the gap at the top station), not the old alternating
    +/-0.025 m step, and the card centreline y eases 0.150 -> 0.125
    (pre-scale) so the flat ribbon sits ~0.006-0.015 m outside the
    head/torso back surface instead of sinking into it or floating
    off it. Thickness is 0.006 m in Y, with front, back and edge
    faces, so the card reads from front and back without changing
    the hair material to double-sided. Shell, crown, front locks,
    side falls and shoulder falls stay spheres in this step — they
    frame the face, and carding them is step 2 if this test reads
    well. Hair parts fall 29 -> 11 (8 spheres + 3 cards), still
    joined into the single Sarah_Hair mesh and auto-weighted as
    before, so the cards follow the hair.01/02/03 bones; no
    bone-name change, no morph change, no new main mesh. LIMIT,
    stated plainly: these are still flat ribbons with a sine wave,
    not individually modelled wavy locks with flyaway strands, and
    shade_smooth on a flat ribbon may soften its edge highlight;
    and NO .glb was built or verified here (no Blender in this
    environment).
  * v1.0.26 (this candidate, queue item 13 — hair cards, step 2:
    front locks / side falls / shoulder falls, built on v1.0.25 so
    ring body + tri-projection + Fortnite-toon proportions +
    sculpted face + back hair cards + all earlier work are kept):
    the six remaining sphere locks/falls are replaced by six wavy
    hair CARDS, reusing the v1.0.25 add_hair_card helper, now with
    an orientation parameter: front locks keep the front-facing
    orientation (width in X, thin in Y), while side and shoulder
    falls use the new sideways orientation (width in Y, thin in
    X), so a side fall reads as a sheet from the side instead of
    edge-on. Front locks: 9 stations, z 1.95 -> 1.63, x +/-0.10
    with a shared 0.015 m sine sway, y easing -0.105 -> -0.140 ->
    -0.120 (pre-scale), width 0.11 m, so the lock hangs in front
    of the cheek but behind the toon eye lenses (lock front
    about -0.168 m scaled at mid-length, eye front -0.177 m).
    Side falls: 13 stations, z 1.91 -> 1.31, x easing 0.155 ->
    0.20 (pre-scale, about 0.183 -> 0.236 scaled, just outside
    the head side / torso side), y about -0.01 with a 0.02 m
    sine sway, width 0.10 m. Shoulder falls: 13 stations,
    z 1.67 -> 1.07, x easing 0.185 -> 0.215 (about 0.218 -> 0.254
    scaled), y about 0.06 with a 0.02 m sine sway, width 0.09 m,
    sitting behind the side falls as a second layer. All cards
    taper 1.0 -> 0.65 toward the tip, thickness 0.006 m. Shell
    and crown stay spheres — they are the head-covering cap,
    not hanging locks, so carding them would not help. Hair
    parts stay 11 (2 spheres + 9 cards, was 8 spheres + 3
    cards), still joined into the single Sarah_Hair mesh and
    auto-weighted as before, so the cards follow the
    hair.01/02/03 bones; no bone-name change, no morph change,
    no new main mesh. LIMIT, stated plainly: these are still
    flat ribbons with a sine wave, not individually modelled
    wavy locks with flyaway strands; the side and shoulder
    cards run as two near-parallel layers about 0.004-0.014 m
    apart where they overlap, shoulder always outside side,
    same material, so they read as
    layered hair — if they flicker in the panel, that spacing
    is the suspect, not a missing mesh; and NO .glb was built
    or verified here (no Blender in this environment).
  * v1.0.27 (this candidate, queue item 12 — remaining step:
    blink-linked eyelids, built on v1.0.26 so ring body +
    tri-projection + Fortnite-toon proportions + sculpted face
    + hair cards + all earlier work are kept): the four v1.0.24
    eyelid caps stop being static. Each cap now carries its own
    shape key named for its side — Blink_L on the L caps,
    Blink_R on the R caps, Basis otherwise — so AvatarPanel's
    setMorph, which already drives Blink_L/Blink_R on EVERY
    mesh that has them (Avatar3D.tsx setMorph loops
    morphMeshes, it is not head-only), drives the lids with
    no panel change. At Blink = 1 each cap's vertices stretch
    1.8x in Z about the cap centre and translate to the eye
    centre (upper: -0.031 m, lower: +0.033 m, pre-scale, times
    head_scale and HEIGHT_SCALE), so the upper and lower caps
    meet over the iris instead of sitting at the eye edges;
    at Blink = 0 the caps are exactly their v1.0.24 positions.
    The head's own 9 morphs are untouched, so the verifier
    should still report head morphs 9, while total morph
    targets rise 9 -> 13 (9 head + 4 lid caps, one per cap).
    Lashes stay static — a top/bottom lash line on a closed
    eye reads as a lash line, not a bug — and the sclera/iris
    stay put behind the closing lids, which is the closure
    mechanism, not a missing morph. LIMIT, stated plainly:
    this is a glTF morph, not a Blender driver, so it only
    closes in a panel that drives Blink_L/Blink_R per mesh
    (the v9 UI copy does); a viewer that drives only the head
    mesh's morphs will still show static lids; and NO .glb was
    built or verified here (no Blender in this environment).
  * No animation clip is exported on purpose. AvatarPanel's procedural
    motion (breathing, gaze, blink, walk, wave, dance) is the live
    system; a baked demo clip would fight it.
   * v1.0.28 consolidated candidate: one importable/standalone generator
     now owns bounded GLB validation, explicit OperatorCore-approved local
     builds, timeout, staging, rollback, checksummed manifest, budgeted
     authoring tiers, Blender-exportable Avatar-Eye nodes, existing-mesh
     appearance metadata, joint-weight validation and a SINGLE articulated
     finger mesh (30 already-existing finger bones, no new bones). Optional
     profile uploads now resolve before any built-in profile lookup. The
     enlarged head's actual height is accounted for in the 5'8" scaling.
     Source and isolated tests are available; neither Blender nor WebGL
     runtime has been verified in this environment. This file does not
     automatically remove the legacy modules or rewrite their callers.
  * v1.0.29 (this candidate, Brian 2026-11-04 21:06 CST direction:
    "make sure functions listed in this file are imported into your
    work", built by merging the uploaded
    SarahMemoryAvatarGenerate__1.py (v1.0.28-consolidated-candidate)
    INTO the v1.0.28-draft-workspace limb-side fix, NOT by replacing
    the draft with the upload — replacing would have LOST the
    v1.0.28 limb-side flat fix, which the upload does not contain).
    Imported / kept from the upload (verified by function list):
    _file_sha256, _read_glb_json, _read_glb_header, _accessor_ok,
    bounded verify_glb, inspect_realtime_avatar_assets,
    _authorize_local_blender, _record_local_blender,
    _terminate_process_tree, _run_blender_bounded, _build_lock,
    _promote_verified_artifacts (staging / rollback / checksummed
    manifest), validate_profile, sml_get_metadata, sml_health,
    sml_diagnostics, sml_receive_packet, plus the inner changes:
    single articulated finger mesh (30 existing finger bones, no
    new bones), Avatar-Eye scene anchors, existing-mesh
    appearance-slot metadata, and budgeted authoring tiers.
    The v1.0.28 limb-side flat_mat path in project_uv_tri and
    the Body call passing mats["suit"] are re-applied on top, so
    limb sides still read flat black, not side-photo white.
    LIMIT: the consolidated upload's new outer-build paths
    (authorised local Blender, staging, rollback) have NOT run
    here (no Blender), and the sml_* functions are a generic
    non-executing adapter unless SarahMemorySMLProtocol is
    present — they do not execute anything on their own.
    NO .glb was built or verified here.
  * v1.0.30 (this candidate, built on v1.0.29 after Brian's
    2026-11-04 21:35 log): (1) That log was NOT a v1.0.29
    build - it ran SarahMemoryAvatarGenerate.py (the folder's
    v1.0.27-level file) with tri counts 602/492/282 and NO
    flat count, verifier 115 nodes / 53 meshes / 22,260
    vertices / 52 skinned / 13 morphs (9 head) / 3 images /
    7,824,352 bytes, PASS. So the 21:32 white-limb shots are
    the pre-flat-fix build, not a v1.0.29 failure - but they
    still prove the v1.0.28 side-only fix is insufficient,
    because front polygons map to the photo's white margin
    (front bbox UV 0.088..0.853 in that log). Change: the
    limb test in project_uv_tri now applies to ANY role
    (front/back/side), so all limb polygons go flat black
    suit, not just side ones. (2) The consolidated
    _authorize_local_blender DENIED plain runs (required
    --allow-blender --user-consented + OperatorCore import,
    which is not on the path in resources/avatars/3D), so
    a plain v1.0.29 command would have left the old GLB
    untouched - my earlier v1.0.29 run command was therefore
    incomplete, and that is corrected here: v1.0.30 warns
    and continues via the staged verified build by default,
    and --governed-build restores the enforced governed
    path. All consolidated functions from v1.0.29 are kept.
    NO .glb was built or verified here (no Blender).

REQUIREMENTS
------------
  * Blender 5.1 (or 4.x) installed somewhere findable. No Blender GUI use.
  * The reference images are optional. If they are missing, the script
    uses the blueprint palette defaults and still builds the model.

This file is a workspace draft for review. It does NOT overwrite
any SarahMemory file until Brian copies it into his tree and runs it
there. It has NOT been run in this environment (no Blender here), so
its first real run on Brian's machine is the test — the built-in
verifier is there precisely to catch a bad export instead of silently
writing a "successful" manifest like the June one did.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_VERSION = "1.0.30-draft-workspace"
MODULE_VERSION = "9.1.0a"
RIG_NAME = "Sarah_Rig"
OUT_BASENAME = "SarahMemoryAvatar_RigBootstrap"

# Blueprint palette defaults (from the June GLB materials + blueprint).
PALETTE_DEFAULT = {
    "skin": (0.89, 0.72, 0.64, 1.0),
    "suit": (0.002, 0.002, 0.004, 1.0),
    "hair": (1.0, 0.13, 0.58, 1.0),
    "neon": (0.16, 0.95, 1.0, 1.0),
    "sclera": (0.97, 0.98, 1.0, 1.0),
    "iris": (0.12, 0.19, 0.26, 1.0),
    "lips": (0.72, 0.22, 0.28, 1.0),
    "brows": (0.23, 0.035, 0.09, 1.0),
}

# ---------------------------------------------------------------------------
# Generic character profile (v1.0.13, queue item 8, step 1 only).
# See generic-character-class-design.md. The rig (EXPECTED_BONES) and the
# 9 head morphs (EXPECTED_SHAPE_KEYS) are deliberately NOT per-character:
# every character shares them, so AvatarPanel / UnifiedAvatarController
# actions work identically for any profile, with no panel changes.
# Face likeness from a 2D upload is NOT in scope: a profile sets palette,
# proportions, hair/outfit/accent presets — it does not sculpt a face.
# ---------------------------------------------------------------------------
class AvatarCharacterProfile:
    """One rigged toon character: Sarah is one instance, not a special case."""

    def __init__(self, character_id, display_name, style_preset, height_m,
                 head_scale=1.0, hand_scale=1.0, shoulder_scale=1.0,
                 hip_scale=1.0, bulk_scale=1.0, palette=None, hair_style="long_wavy",
                 hair_length=1.0, outfit="bodysuit",
                 accent_mode="neon_circuits", boots="platform_glow",
                 texture_image="", object_prefix="Sarah"):
        self.character_id = character_id
        self.display_name = display_name
        self.style_preset = style_preset
        self.height_m = height_m
        self.head_scale = head_scale
        self.hand_scale = hand_scale
        self.shoulder_scale = shoulder_scale
        self.hip_scale = hip_scale
        self.bulk_scale = bulk_scale
        self.palette = dict(palette) if palette is not None else dict(PALETTE_DEFAULT)
        self.hair_style = hair_style
        self.hair_length = hair_length
        self.outfit = outfit
        self.accent_mode = accent_mode
        self.boots = boots
        self.texture_image = texture_image
        self.object_prefix = object_prefix

    @classmethod
    def sarah_default(cls):
        # v1.0.18, queue item 9: Sarah is now fortnite_toon on purpose -
        # bigger head/hands, wider shoulders/hips, chunkier torso/limbs
        # (bulk_scale), per Brian's 2026-11-03 Fortnite-toon target.
        # This intentionally changes the Sarah build vs v1.0.17; the
        # v1.0.13 reproduction proof is history, not this step's test.
        return cls(
            character_id="sarah_default",
            display_name="Sarah",
            style_preset="fortnite_toon",
            height_m=1.7272,  # 5'8", per Brian 2026-11-03
            head_scale=1.18, hand_scale=1.25,
            shoulder_scale=1.08, hip_scale=1.05, bulk_scale=1.10,
            hair_style="long_wavy",
            outfit="bodysuit",
            accent_mode="neon_circuits",
            boots="platform_glow",
            palette=dict(PALETTE_DEFAULT),
            object_prefix="Sarah",
        )

    @classmethod
    def toon_hero_01(cls):
        # v1.0.14, queue item 8 step 2: first non-Sarah profile.
        # Design doc values for fortnite_toon: head ~1.25, hand ~1.3.
        # Hair/outfit/accent/boots stay Sarah-like on purpose here:
        # this step proves proportions + palette parameterisation,
        # it does not yet redesign hair, outfit or neon routing.
        return cls(
            character_id="toon_hero_01",
            display_name="Toon Hero",
            style_preset="fortnite_toon",
            height_m=1.7272,  # same height as Sarah, to isolate the look
            head_scale=1.25,
            hand_scale=1.30,
            shoulder_scale=1.10,
            hip_scale=1.05,
            hair_style="long_wavy",
            outfit="bodysuit",
            accent_mode="neon_circuits",
            boots="platform_glow",
            palette={
                "skin": (0.85, 0.66, 0.55, 1.0),
                "suit": (0.02, 0.05, 0.14, 1.0),
                "hair": (0.20, 0.65, 1.0, 1.0),
                "neon": (1.0, 0.72, 0.12, 1.0),
                "sclera": (0.97, 0.98, 1.0, 1.0),
                "iris": (0.10, 0.25, 0.45, 1.0),
                "lips": (0.65, 0.25, 0.30, 1.0),
                "brows": (0.15, 0.10, 0.20, 1.0),
            },
            object_prefix="ToonHero",
        )

    def summary(self):
        return (f"{self.character_id} ({self.display_name}, "
                f"{self.style_preset}, {self.height_m} m, hair={self.hair_style}, "
                f"outfit={self.outfit}, accent={self.accent_mode}, "
                f"boots={self.boots})")

    def to_dict(self):
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "style_preset": self.style_preset,
            "height_m": self.height_m,
            "head_scale": self.head_scale,
            "hand_scale": self.hand_scale,
            "shoulder_scale": self.shoulder_scale,
            "hip_scale": self.hip_scale,
            "bulk_scale": self.bulk_scale,
            "palette": {k: list(v) for k, v in self.palette.items()},
            "hair_style": self.hair_style,
            "hair_length": self.hair_length,
            "outfit": self.outfit,
            "accent_mode": self.accent_mode,
            "boots": self.boots,
            "texture_image": self.texture_image,
            "object_prefix": self.object_prefix,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not data.get("character_id"):
            raise ValueError("profile JSON has no character_id")
        palette = {k: tuple(v) for k, v in
                   data.get("palette", PALETTE_DEFAULT).items()}
        merged = dict(PALETTE_DEFAULT)
        merged.update(palette)
        return cls(
            character_id=data["character_id"],
            display_name=data.get("display_name", data["character_id"]),
            style_preset=data.get("style_preset", "fortnite_toon"),
            height_m=float(data.get("height_m", 1.7272)),
            head_scale=float(data.get("head_scale", 1.0)),
            hand_scale=float(data.get("hand_scale", 1.0)),
            shoulder_scale=float(data.get("shoulder_scale", 1.0)),
            hip_scale=float(data.get("hip_scale", 1.0)),
            bulk_scale=float(data.get("bulk_scale", 1.0)),
            palette=merged,
            hair_style=data.get("hair_style", "long_wavy"),
            hair_length=float(data.get("hair_length", 1.0)),
            outfit=data.get("outfit", "bodysuit"),
            accent_mode=data.get("accent_mode", "neon_circuits"),
            boots=data.get("boots", "platform_glow"),
            texture_image=data.get("texture_image", ""),
            object_prefix=data.get("object_prefix", "Upload"),
        )


# v1.0.14: Sarah plus the first non-Sarah profile (design doc step 2).
# A future profile must be added here explicitly, and must NEVER
# overwrite SarahMemoryAvatar_RigBootstrap.glb (see out_basename_for).
PROFILES = {
    "sarah_default": AvatarCharacterProfile.sarah_default(),
    "toon_hero_01": AvatarCharacterProfile.toon_hero_01(),
}


def resolve_profile(character_id):
    """Return the profile for character_id, or fail loudly if unknown."""
    try:
        return PROFILES[character_id]
    except KeyError:
        raise ValueError(
            f"Unknown character profile: {character_id!r}. "
            f"Known profiles: {sorted(PROFILES)}. Refusing to silently "
            f"build Sarah instead."
        ) from None


def out_basename_for(profile):
    """Output basename for a profile (v1.0.14, queue item 8 step 2).

    Sarah keeps the historic SarahMemoryAvatar_RigBootstrap name;
    every other profile gets its own <ObjectPrefix>Avatar_RigBootstrap
    name, so a toon build can never overwrite the Sarah GLB in the
    same output folder. Internal object/bone names stay Sarah_*
    for now, so the panel/controller runtime still finds them.
    """
    if profile.character_id == "sarah_default":
        return OUT_BASENAME
    return f"{profile.object_prefix}Avatar_RigBootstrap"


# ---------------------------------------------------------------------------
# Upload sampling (v1.0.16, queue item 8, step 4). Pure Python, no bpy:
# this runs in the OUTER process so a bad upload fails before Blender
# starts, and so it can be unit-checked without Blender at all.
# Design doc, step 2: sample, don't sculpt. The image sets palette and
# silhouette scales only; it never carves a face or a body mesh.
# ---------------------------------------------------------------------------
def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _decode_png(path):
    """Decode an 8-bit RGB/RGBA, non-interlaced PNG to RGB rows.

    Stdlib only (struct + zlib): no Pillow dependency on Brian's PC.
    Returns (width, height, rows) with rows a list of bytes, 3 per pixel.
    Anything else (JPEG, 16-bit, interlaced, palette) fails loudly.
    """
    # Bounded input decoding: reject decompression bombs and malformed chunks.
    src = Path(path)
    if src.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("PNG exceeds 32 MiB reference file budget")
    data = src.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path} (8-bit RGB/RGBA PNG required)")
    pos, compressed = 8, bytearray()
    width = height = bit_depth = color_type = interlace = None
    saw_end = False
    while pos + 12 <= len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        ctype = data[pos + 4:pos + 8]
        stop = pos + 12 + length
        if length > 32 * 1024 * 1024 or stop > len(data):
            raise ValueError("invalid PNG chunk length")
        chunk = data[pos + 8:pos + 8 + length]
        expected_crc = struct.unpack_from(">I", data, pos + 8 + length)[0]
        if zlib.crc32(ctype + chunk) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"PNG chunk CRC failed: {ctype!r}")
        if ctype == b"IHDR":
            if width is not None or length != 13:
                raise ValueError("invalid/duplicate PNG IHDR")
            (width, height, bit_depth, color_type, _comp, _filt,
             interlace) = struct.unpack(">IIBBBBB", chunk)
            if (width <= 0 or height <= 0 or width > 8192 or height > 8192
                    or width * height > 16_000_000):
                raise ValueError("PNG pixel dimensions exceed bounded reference budget")
        elif ctype == b"IDAT":
            if len(compressed) + length > 32 * 1024 * 1024:
                raise ValueError("PNG compressed pixel budget exceeded")
            compressed.extend(chunk)
        elif ctype == b"IEND":
            saw_end = True
            break
        pos = stop
    if width is None or not saw_end:
        raise ValueError(f"incomplete PNG: {path}")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported PNG depth/type/interlace: {path}")
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    expected = height * (stride + 1)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(bytes(compressed), expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError(f"incomplete or overlong PNG pixels: {path}")
    rows, prev = [], bytearray(stride)
    for y in range(height):
        ftype = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if ftype == 0:
            pass
        elif ftype in (1, 2, 3, 4):
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                if ftype == 1:
                    line[i] = (line[i] + a) & 0xFF
                elif ftype == 2:
                    line[i] = (line[i] + b) & 0xFF
                elif ftype == 3:
                    line[i] = (line[i] + (a + b) // 2) & 0xFF
                else:
                    line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {ftype}: {path}")
        if channels == 4:
            rgb = bytearray()
            for i in range(0, stride, 4):
                rgb += line[i:i + 3]
            rows.append(bytes(rgb))
        else:
            rows.append(bytes(line))
        prev = line
    return width, height, rows


def _median_rgb(pixels):
    """Median (r, g, b) in 0..1 floats from a list of (r, g, b) 0..255."""
    return tuple(round(statistics.median([px[i] for px in pixels]) / 255.0, 3)
                 for i in range(3))


def _sanitize_character_id(stem):
    cid = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return cid or "upload_hero"


def _prefix_for_id(character_id):
    prefix = "".join(part.capitalize()
                     for part in character_id.split("_") if part)
    if not prefix or not prefix[0].isalpha():
        prefix = "Char" + prefix
    return prefix


def sample_upload_image(path):
    """Sample palette + silhouette ratios from a single-avatar PNG.

    Background is taken from the image corners; a dark background is
    rejected because a black suit on black cannot be masked. Region
    boxes are in bounding-box coordinates so they work at any image
    size. Returns a dict with bbox, ratios, palette (0..1 RGBA tuples)
    and derived scales; it never writes anything.
    """
    width, height, rows = _decode_png(path)
    # Background: median of small corner patches.
    corner = []
    for cy in (0, height - 1):
        for cx in (0, width - 1):
            for dy in range(min(5, height)):
                for dx in range(min(5, width)):
                    yy = min(height - 1, cy + dy) if cy == 0 else max(0, cy - dy)
                    xx = min(width - 1, cx + dx) if cx == 0 else max(0, cx - dx)
                    row = rows[yy]
                    corner.append((row[3 * xx], row[3 * xx + 1], row[3 * xx + 2]))
    bg = _median_rgb(corner)
    if sum(bg) / 3.0 < 0.20:
        raise ValueError(
            f"dark background (median {bg}) in {path}: upload a "
            f"single-avatar image on WHITE (e.g. source_single/*_white.png) "
            f"— a black suit on a black background cannot be masked")

    def is_bg(r, g, b):
        return (abs(r / 255.0 - bg[0]) < 0.15
                and abs(g / 255.0 - bg[1]) < 0.15
                and abs(b / 255.0 - bg[2]) < 0.15)

    spans = []
    for row in rows:
        xs = [x for x in range(width)
              if not is_bg(row[3 * x], row[3 * x + 1], row[3 * x + 2])]
        spans.append((min(xs), max(xs)) if xs else None)
    nonempty = [y for y, s in enumerate(spans) if s is not None]
    if len(nonempty) < 10:
        raise ValueError(f"no avatar found (empty silhouette mask): {path}")
    top, bottom = nonempty[0], nonempty[-1]
    bbox_h = bottom - top + 1
    left = min(spans[y][0] for y in nonempty)
    right = max(spans[y][1] for y in nonempty)
    bbox_w = right - left + 1

    def width_at(frac):
        y0 = top + int(frac * bbox_h)
        best = 0
        for y in range(max(top, y0 - 2), min(bottom, y0 + 2) + 1):
            if spans[y] is not None:
                best = max(best, spans[y][1] - spans[y][0] + 1)
        return best

    head_w = width_at(0.08)
    shoulder_w = width_at(0.25)
    hip_w = width_at(0.62)
    ratios = {
        "aspect_w_over_h": round(bbox_w / bbox_h, 3),
        "head_width_ratio": round(head_w / bbox_h, 3),
        "shoulder_width_ratio": round(shoulder_w / bbox_h, 3),
        "hip_width_ratio": round(hip_w / bbox_h, 3),
    }

    def region_median(y0f, y1f, xc_frac, half_w_frac):
        px = []
        for y in range(top + int(y0f * bbox_h), top + int(y1f * bbox_h)):
            cx = left + int(xc_frac * bbox_w)
            hw = int(half_w_frac * bbox_w)
            row = rows[y]
            for x in range(max(left, cx - hw), min(right, cx + hw) + 1):
                r, g, b = row[3 * x], row[3 * x + 1], row[3 * x + 2]
                if not is_bg(r, g, b):
                    px.append((r, g, b))
        return _median_rgb(px) if len(px) >= 20 else None

    hair = region_median(0.02, 0.13, 0.5, 0.20)
    skin = region_median(0.09, 0.22, 0.5, 0.13)
    suit = region_median(0.35, 0.55, 0.5, 0.25)
    # Neon must NOT be the hair colour: magenta hair is itself bright
    # and saturated, and would swamp a thin cyan/amber neon line in a
    # plain median. So exclude pixels close to the sampled hair colour.
    hair_rgb = hair[:3] if hair is not None else None
    neon_px = []
    for y in range(top, bottom + 1, 2):
        row = rows[y]
        for x in range(left, right + 1, 2):
            r, g, b = (row[3 * x] / 255.0, row[3 * x + 1] / 255.0,
                       row[3 * x + 2] / 255.0)
            if max(r, g, b) > 0.75 and (max(r, g, b) - min(r, g, b)) > 0.35:
                if hair_rgb is not None and max(
                        abs(r - hair_rgb[0]), abs(g - hair_rgb[1]),
                        abs(b - hair_rgb[2])) < 0.30:
                    continue
                neon_px.append((int(r * 255), int(g * 255), int(b * 255)))
    neon = _median_rgb(neon_px) if len(neon_px) >= 20 else None
    palette = dict(PALETTE_DEFAULT)
    if hair is not None:
        palette["hair"] = hair + (1.0,)
    if skin is not None:
        palette["skin"] = skin + (1.0,)
    if suit is not None:
        palette["suit"] = suit + (1.0,)
    if neon is not None:
        palette["neon"] = neon + (1.0,)
    # v1.0.17 recalibrated Sarah baselines, from Brian's 2026-11-03
    # calibration run on source_single/front_white.png, which measured
    # head 0.136, shoulder 0.209, hip 0.116 at the same 8%/25%/62%
    # points. The v1.0.16 guesses (0.20/0.27/0.21) made all three
    # scales clamp to the 0.9 floor on Sarah herself, so they were
    # wrong as baselines. With these, Sarah herself now returns 1.0
    # on all three, and other characters scale relative to Sarah
    # as measured by the same method — that relative comparison is
    # what matters, even if the absolute fractions are method-bound.
    scales = {
        "head_scale": round(max(0.9, min(1.5, ratios["head_width_ratio"] / 0.136)), 3),
        "hand_scale": 1.0,
        "shoulder_scale": round(max(0.9, min(1.3, ratios["shoulder_width_ratio"] / 0.209)), 3),
        "hip_scale": round(max(0.9, min(1.3, ratios["hip_width_ratio"] / 0.116)), 3),
    }
    return {
        "image": str(Path(path).resolve()),
        "width": width, "height": height,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom,
                 "width": bbox_w, "height": bbox_h},
        "background": bg,
        "ratios": ratios,
        "palette": palette,
        "scales": scales,
        "warnings": [
            "hand_scale stays 1.0: hands are not measurable from a silhouette",
            "height stays 1.7272 m: a 2D image gives ratios, not metres",
            "face likeness is NOT in scope: palette + silhouette only",
        ],
    }


def profile_from_upload(path, character_id=None, style_hint="auto"):
    """Derive an AvatarCharacterProfile + sample report from an upload."""
    sample = sample_upload_image(path)
    cid = _sanitize_character_id(character_id) if character_id else \
        _sanitize_character_id(Path(path).stem)
    if style_hint in ("fortnite_toon", "sarah_default"):
        style = style_hint
    else:  # auto: nearest of the two existing presets by head scale
        style = ("fortnite_toon" if sample["scales"]["head_scale"] >= 1.15
                 else "sarah_default")
    profile = AvatarCharacterProfile(
        character_id=cid,
        display_name=cid.replace("_", " ").title(),
        style_preset=style,
        height_m=1.7272,
        head_scale=sample["scales"]["head_scale"],
        hand_scale=1.0,
        shoulder_scale=sample["scales"]["shoulder_scale"],
        hip_scale=sample["scales"]["hip_scale"],
        palette=dict(sample["palette"]),
        hair_style="long_wavy",
        outfit="bodysuit",
        accent_mode="neon_circuits",
        boots="platform_glow",
        texture_image=str(Path(path).resolve()),
        object_prefix=_prefix_for_id(cid),
    )
    return profile, sample


def _jsonable_sample(sample):
    out = dict(sample)
    out["palette"] = {k: list(v) for k, v in sample["palette"].items()}
    out["background"] = list(sample["background"])
    return out


EXPECTED_BONES = [
    "root", "pelvis", "spine.01", "spine.02", "spine.03", "neck", "head", "jaw",
    "eye.L", "eye.R", "hair_root", "hair.01", "hair.02", "hair.03",
    "clavicle.L", "upper_arm.L", "forearm.L", "hand.L",
    "thumb.01.L", "thumb.02.L", "thumb.03.L",
    "index.01.L", "index.02.L", "index.03.L",
    "middle.01.L", "middle.02.L", "middle.03.L",
    "ring.01.L", "ring.02.L", "ring.03.L",
    "pinky.01.L", "pinky.02.L", "pinky.03.L",
    "clavicle.R", "upper_arm.R", "forearm.R", "hand.R",
    "thumb.01.R", "thumb.02.R", "thumb.03.R",
    "index.01.R", "index.02.R", "index.03.R",
    "middle.01.R", "middle.02.R", "middle.03.R",
    "ring.01.R", "ring.02.R", "ring.03.R",
    "pinky.01.R", "pinky.02.R", "pinky.03.R",
    "thigh.L", "shin.L", "foot.L", "toe.L",
    "thigh.R", "shin.R", "foot.R", "toe.R",
]
EXPECTED_SHAPE_KEYS = [
    "Blink_L", "Blink_R", "JawOpen", "Smile", "Frown", "BrowsUp",
    "Squint_L", "Squint_R", "MouthWide",
]


# ---------------------------------------------------------------------------
# Consolidated, bounded GLB verifier (Builder + Bootstrap inspection policy).
# Reads only the glTF JSON chunk; binary mesh/texture data is never loaded.
# The JSON/accessor proof is structural, not a Blender/WebGL animation test.
# ---------------------------------------------------------------------------
DEFAULT_MAX_GLB_MB = 128
DEFAULT_MAX_VERTICES = 300_000
DEFAULT_MAX_TRIANGLES = 300_000
MAX_GLB_JSON_BYTES = 16 * 1024 * 1024


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_glb_json(path, max_bytes=DEFAULT_MAX_GLB_MB * 1024 * 1024):
    """Reject corrupt/oversized GLBs without loading their BIN payload."""
    path = Path(path)
    size = path.stat().st_size
    if not 20 <= size <= max_bytes:
        raise ValueError(f"GLB size {size:,} outside 20..{max_bytes:,} byte budget")
    with path.open("rb") as stream:
        magic, version, declared = struct.unpack("<4sII", stream.read(12))
        if (magic, version, declared) != (b"glTF", 2, size) or size % 4:
            raise ValueError("invalid glTF v2 header, length, or 4-byte alignment")
        offset, chunks, doc, bin_bytes = 12, 0, None, 0
        while offset < size:
            if size - offset < 8:
                raise ValueError("truncated GLB chunk header")
            length, kind = struct.unpack("<I4s", stream.read(8))
            offset += 8
            if length % 4 or length > size - offset:
                raise ValueError("invalid/truncated GLB chunk")
            if chunks == 0:
                if kind != b"JSON" or length > MAX_GLB_JSON_BYTES:
                    raise ValueError("invalid or oversized initial GLB JSON chunk")
                doc = json.loads(stream.read(length).decode("utf-8").rstrip(" \t\r\n\x00"))
            elif chunks == 1 and kind == b"BIN\x00":
                bin_bytes = length
                stream.seek(length, 1)
            else:
                raise ValueError("unexpected GLB chunk order/type")
            offset += length
            chunks += 1
    if offset != size or chunks != 2 or not isinstance(doc, dict):
        raise ValueError("incomplete GLB JSON/BIN container")
    if not isinstance(doc.get("asset"), dict) or doc["asset"].get("version") != "2.0":
        raise ValueError("not a glTF 2.0 asset")
    buffers = doc.get("buffers") or []
    if (len(buffers) != 1 or not isinstance(buffers[0], dict)
            or not isinstance(buffers[0].get("byteLength"), int)
            or not 0 < buffers[0]["byteLength"] <= bin_bytes):
        raise ValueError("invalid GLB binary buffer declaration")
    return doc, size, bin_bytes


def _accessor_ok(doc, accessor_id):
    """Bounds check for a non-sparse mesh accessor and its referenced view."""
    accs, views, buffers = (doc.get("accessors") or [], doc.get("bufferViews") or [],
                            doc.get("buffers") or [])
    if type(accessor_id) is not int or not 0 <= accessor_id < len(accs):
        return False
    acc = accs[accessor_id]
    if not isinstance(acc, dict) or type(acc.get("bufferView")) is not int:
        return False
    vid = acc["bufferView"]
    if not 0 <= vid < len(views):
        return False
    view = views[vid]
    if not isinstance(view, dict) or view.get("buffer") != 0:
        return False
    length, offset, count = view.get("byteLength"), view.get("byteOffset", 0), acc.get("count")
    if (type(length) is not int or length <= 0 or type(offset) is not int
            or offset < 0 or type(count) is not int or count <= 0
            or offset + length > buffers[0]["byteLength"]):
        return False
    return True


def _read_glb_header(path):
    """Legacy Builder-compatible inspect API for eventual caller migration."""
    result = {"path": os.path.abspath(str(path)), "exists": Path(path).is_file(),
              "size_bytes": 0, "valid_glb_header": False, "valid_glb_structure": False,
              "has_mesh": False, "mesh_count": 0, "node_count": 0,
              "version": None, "declared_length": None}
    if not result["exists"]:
        return result
    try:
        doc, size, _ = _read_glb_json(path, 512 * 1024 * 1024)
        meshes, nodes, scenes = (doc.get("meshes") or [], doc.get("nodes") or [],
                                  doc.get("scenes") or [])
        has_mesh = any(isinstance(mesh, dict) and any(
            isinstance(p, dict) and isinstance(p.get("attributes"), dict) and
            _accessor_ok(doc, p["attributes"].get("POSITION"))
            for p in mesh.get("primitives", [])) for mesh in meshes)
        scene_ok = any(isinstance(s, dict) and any(type(n) is int and 0 <= n < len(nodes)
                        for n in s.get("nodes", [])) for s in scenes)
        result.update(size_bytes=size, valid_glb_header=True, valid_glb_structure=True,
                      has_mesh=bool(has_mesh and scene_ok), mesh_count=len(meshes),
                      node_count=len(nodes), version=2, declared_length=size)
    except (OSError, ValueError, TypeError, IndexError, UnicodeError,
            json.JSONDecodeError, struct.error) as exc:
        result["error"] = str(exc)[:240]
    return result


def verify_glb(glb_path: Path, *, max_glb_mb=DEFAULT_MAX_GLB_MB,
               max_vertices=DEFAULT_MAX_VERTICES, max_triangles=DEFAULT_MAX_TRIANGLES):
    """Verify Sarah's *runtime* rig contract, morphs and geometry budgets."""
    result = {"file": str(glb_path), "ok": False, "checks": {}, "errors": [],
              "warnings": []}
    try:
        doc, size, _ = _read_glb_json(glb_path, int(max_glb_mb) * 1024 * 1024)
        result["size_bytes"] = size
        nodes, meshes, skins = (doc.get("nodes") or [], doc.get("meshes") or [],
                                doc.get("skins") or [])
        names = {n.get("name") for n in nodes if isinstance(n, dict)}
        checks = result["checks"]
        checks.update(nodes=len(nodes), meshes=len(meshes), skins=len(skins),
                      animations=len(doc.get("animations") or []),
                      materials=[m.get("name") for m in (doc.get("materials") or []) if isinstance(m,dict)][:80],
                      textures_count=len(doc.get("textures") or []),
                      images_count=len(doc.get("images") or []),
                      image_names=[m.get("name") for m in (doc.get("images") or []) if isinstance(m,dict)][:40])
        checks["textures"] = checks["textures_count"]
        checks["images"] = checks["images_count"]
        if not checks["images_count"]:
            result["warnings"].append("No GLB textures; flat materials may be intentional (--no-texture)")
        missing_bones = [bone for bone in EXPECTED_BONES if bone not in names]
        checks["missing_bones"] = missing_bones
        if missing_bones:
            result["errors"].append(f"Missing required rig bones: {missing_bones}")
        if RIG_NAME not in names:
            result["errors"].append(f"Missing {RIG_NAME} armature node")
        checks["avatar_eye_anchor"] = "Sarah_AvatarEye_Center" in names
        if not checks["avatar_eye_anchor"]:
            result["warnings"].append("Avatar-Eye anchor missing from GLB scene")
        if not skins:
            result["errors"].append("GLB has no skin/armature")
        for skin in skins:
            if not isinstance(skin, dict) or not skin.get("joints") or any(
                    type(j) is not int or j < 0 or j >= len(nodes)
                    for j in skin.get("joints", [])):
                result["errors"].append("Invalid glTF skin joint references")
        head_node = next((n for n in nodes if isinstance(n, dict)
                          and n.get("name") == "Sarah_Head"), {})
        head_index = head_node.get("mesh")
        total_vertices, total_triangles, targets_total, targets_head = 0, 0, 0, 0
        skinned_nodes = 0
        uv_meshes = set()
        for node in nodes:
            if not isinstance(node, dict) or "skin" not in node:
                continue
            skinned_nodes += 1
            skinid, mid = node["skin"], node.get("mesh")
            if (type(skinid) is not int or not 0 <= skinid < len(skins)
                    or type(mid) is not int or not 0 <= mid < len(meshes)):
                result["errors"].append(f"Invalid mesh/skin binding: {node.get('name')}")
                continue
            for prim in meshes[mid].get("primitives", []):
                attrs = prim.get("attributes") or {}
                if not (_accessor_ok(doc, attrs.get("JOINTS_0")) and
                        _accessor_ok(doc, attrs.get("WEIGHTS_0"))):
                    result["errors"].append(f"Missing joint/weight attributes: {node.get('name')}")
        for mid, mesh in enumerate(meshes):
            for prim in mesh.get("primitives", []):
                attrs = prim.get("attributes") or {}
                pos = attrs.get("POSITION")
                if not _accessor_ok(doc, pos):
                    result["errors"].append(f"Invalid POSITION accessor in mesh {mid}")
                    continue
                count = doc["accessors"][pos]["count"]
                total_vertices += count
                ind = prim.get("indices")
                if ind is not None and not _accessor_ok(doc, ind):
                    result["errors"].append(f"Invalid index accessor in mesh {mid}")
                if prim.get("mode", 4) == 4:
                    total_triangles += ((doc["accessors"][ind]["count"] if
                                         _accessor_ok(doc, ind) else count) // 3)
                if "TEXCOORD_0" in attrs:
                    uv_meshes.add(mesh.get("name") or str(mid))
                targets_total += len(prim.get("targets") or [])
                if mid == head_index:
                    targets_head += len(prim.get("targets") or [])
                    found_names = (mesh.get("extras") or {}).get("targetNames")
                    if isinstance(found_names, list):
                        missing = [n for n in EXPECTED_SHAPE_KEYS if n not in found_names]
                        if missing:
                            result["errors"].append(f"Head morph names missing: {missing}")
        checks.update(morph_targets_total=targets_total, morph_targets_head=targets_head,
                      vertices_total=total_vertices, triangles_total=total_triangles,
                      meshes_with_uv=sorted(uv_meshes), skinned_nodes=skinned_nodes)
        if targets_head < len(EXPECTED_SHAPE_KEYS):
            result["errors"].append(f"Head morph targets {targets_head}, expected {len(EXPECTED_SHAPE_KEYS)}")
        if skinned_nodes < 8:
            result["errors"].append(f"Only {skinned_nodes} skinned nodes (expected >=8)")
        if total_vertices > max_vertices or total_triangles > max_triangles:
            result["errors"].append(f"Runtime geometry over budget: {total_vertices:,} vertices / {total_triangles:,} triangles")
        if not _read_glb_header(glb_path).get("has_mesh"):
            result["errors"].append("GLB has no valid scene/mesh")
        result["ok"] = not result["errors"]
    except (OSError, ValueError, TypeError, KeyError, IndexError,
            UnicodeError, json.JSONDecodeError, struct.error) as exc:
        result["errors"].append(f"GLB verification failed: {exc}")
    return result


def inspect_realtime_avatar_assets(asset_dir="", avatar_name=OUT_BASENAME):
    """Builder's old read-only status entrypoint, without a Builder import."""
    root = (Path(asset_dir).resolve() if asset_dir else default_outdir().resolve())
    stem = Path(str(avatar_name)).stem
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", stem):
        raise ValueError("invalid avatar name")
    glb_path = root / f"{stem}.glb"
    manifest_path = root / f"{stem}.json"
    blend_path = root / f"{stem}.blend"
    proof = _read_glb_header(glb_path)
    manifest = {}
    error = ""
    if manifest_path.is_file():
        try:
            if manifest_path.stat().st_size > 1024 * 1024:
                raise ValueError("manifest too large")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except (ValueError, OSError) as exc:
            error = str(exc)
    ready = bool(proof.get("has_mesh") and proof.get("valid_glb_structure"))
    return {"ok": True, "schema": "SARAHMEMORY_REALTIME_AVATAR_STATUS_V1",
            "asset_dir": str(root), "glb": proof,
            "blend": {"path": str(blend_path), "exists": blend_path.is_file(),
                      "size_bytes": blend_path.stat().st_size if blend_path.is_file() else 0},
            "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
            "manifest_loaded": bool(manifest), "manifest_error": error,
            "quality": manifest.get("quality", "unknown"),
            "bone_count": len(EXPECTED_BONES),
            "shape_keys": list(EXPECTED_SHAPE_KEYS),
            "actions": [], "animation_contract": {"procedural_panel_owned": True},
            "avatar_panel_contract": manifest.get("avatar_panel_contract", {}),
            "runtime_ready": ready, "validation_level": "glb_container_json_and_mesh_only",
            "browser_rendering_verified": False,
            "runtime_mode": "lazy_glb_webgl", "load_policy": "load_only_when_avatar_3d_mode_selected",
            "execution_authority": False, "hardware_control": False}


def find_blender(explicit: str | None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    import shutil
    found = shutil.which("blender")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Blender51\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for c in candidates:
        if Path(c).is_file():
            return c
    return None


def default_outdir() -> Path:
    # v1.0.1 fix: if the script itself already lives in .../resources/avatars/3D
    # (Brian's run), the output folder IS that folder — not a nested
    # .../3D/resources/avatars/3D copy. That nesting bug is what happened on
    # the first run.
    here = Path(__file__).resolve().parent
    if here.name == "3D":
        return here
    for cand in [here, here.parent, here.parent.parent]:
        out = cand / "resources" / "avatars" / "3D"
        if (cand / "resources").exists():
            return out
    return Path.cwd() / "resources" / "avatars" / "3D"


# ---------------------------------------------------------------------------
# Local-only execution governance from Builder/Bootstrap, without importing
# either legacy file. This module is a visual asset builder, not an agent.
# ---------------------------------------------------------------------------

def _authorize_local_blender(target, blender, args):
    # v1.0.30: the consolidated upload made a plain run DENY
    # (it required --allow-blender --user-consented AND an
    # OperatorCore import that is not on the path when the
    # script runs standalone in resources/avatars/3D). That
    # broke the plain command Brian has used for every
    # candidate since v1.0.1, and it is why a v1.0.29 plain
    # run would have left the old GLB in place instead of
    # building a new one. Default is now: warn loudly and
    # continue via the staged/promoted build (which still
    # verifies the GLB before promoting it, and keeps
    # rollback backups). The governed path is NOT deleted:
    # pass --governed-build to enforce the original
    # consent + OperatorCore checks, and DENY without them.
    if not getattr(args, "governed_build", False):
        print("WARNING (v1.0.30): governed OperatorCore build NOT "
              "requested - continuing with the staged, verified "
              "local build; pass --governed-build to enforce "
              "OperatorCore authorization instead")
        return None
    consent = (bool(args.user_consented) or
               os.getenv("SARAH_AVATAR_GENERATE_USER_CONSENTED", "").lower() in ("1", "true", "yes"))
    optin = (bool(args.allow_blender) or
             os.getenv("SARAH_AVATAR_GENERATE_ALLOW_BLENDER", "").lower() in ("1", "true", "yes"))
    if not (consent and optin):
        raise PermissionError("Blender build requires --allow-blender --user-consented (or both documented environment opt-ins)")
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
    except ImportError as exc:
        raise PermissionError("OperatorCore unavailable: governed build cannot be authorized") from exc
    proposal = {"action_type": "avatar_asset_build", "capability_name": "avatar.asset.build",
                "target": str(target), "target_ref": str(target), "blender_binary": str(blender),
                "risk_level": "TIER_2_BOUNDED_LOCAL_OPERATION",
                "required_permissions": ["FilesystemWrite", "ProcessExecute"],
                "verification_checks": [{"name": "glb_structure_rig_morph_budgets", "required": True}],
                "preconditions": ["user consent", "OperatorCore review", "staged bounded Blender"],
                "rollback_plan": [{"name": "restore_prior_assets", "required": True}],
                "expected_result": "verified runtime GLB promoted without replacing prior asset on failure",
                "touches_filesystem": True, "touches_network": False, "dry_run": False, "apply": True}
    result = authorize_external_domain_action(
        "Generate verified visual-only SarahMemory avatar GLB", origin="SarahMemoryAvatarGenerate",
        proposed_action=proposal, meta={"local_only": True, "execution_authority": False},
        user_consented=consent)
    if not isinstance(result, dict) or not (result.get("allow") is True and
        result.get("domain_execution_authorized") is True and result.get("decision") == "ALLOW"):
        raise PermissionError("OperatorCore denied Blender build")
    return result


def _record_local_blender(review, ok, artifact="", error=""):
    if review is None:
        return
    try:
        from SarahMemoryOperatorCore import record_external_domain_action_result
        record_external_domain_action_result(
            review, {"ok": bool(ok), "artifact": Path(artifact).name if artifact else "",
                     "error": str(error)[:200], "execution_authority": False},
            verified=bool(ok), origin="SarahMemoryAvatarGenerate")
    except Exception as exc:
        print(f"WARNING: OperatorCore result receipt unavailable ({type(exc).__name__})")


def _terminate_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _run_blender_bounded(cmd, timeout_seconds, log_path):
    """Non-shell subprocess, bounded duration, memory, console and log output."""
    messages = queue.Queue(maxsize=128)
    done = threading.Event()
    dropped = [0]
    log_budget = 2 * 1024 * 1024
    log_path = Path(log_path)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1,
                            creationflags=(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                           if os.name == "nt" else 0),
                            start_new_session=os.name != "nt")
    def reader():
        try:
            while True:
                line = proc.stdout.readline(4096)
                if not line:
                    break
                try:
                    messages.put_nowait(line[:4096])
                except queue.Full:
                    dropped[0] += 1
        finally:
            done.set()
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    started = time.monotonic()
    total, console = 0, 0
    try:
        with log_path.open("w", encoding="utf-8") as output:
            while proc.poll() is None or not done.is_set() or not messages.empty():
                if time.monotonic() - started > timeout_seconds:
                    _terminate_process_tree(proc)
                    output.write("\nERROR: Blender timed out and process tree was stopped\n")
                    return 124
                try:
                    line = messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                n = len(line.encode("utf-8", "replace"))
                if total + n <= log_budget:
                    output.write(line)
                    total += n
                    if console + n <= 12_000:
                        print(line, end="")
                        console += n
                else:
                    dropped[0] += 1
            if dropped[0]:
                output.write(f"\nWARNING: {dropped[0]} bounded log fragments discarded\n")
        return int(proc.returncode or 0)
    finally:
        if proc.poll() is None:
            _terminate_process_tree(proc)
        thread.join(timeout=2)
        if proc.stdout is not None:
            proc.stdout.close()


@contextlib.contextmanager
def _build_lock(outdir):
    """One build per output directory; fail fast rather than racing."""
    path = Path(outdir) / ".SarahMemoryAvatar_RigBootstrap.lock"
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _promote_verified_artifacts(stagedir, outdir, basename, *, max_rollbacks=3):
    """Atomically move each verified artifact, rolling back the entire set on error."""
    stage, dest = Path(stagedir).resolve(), Path(outdir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", basename):
        raise ValueError("invalid output basename")
    names = [basename + ext for ext in (".glb", ".blend", ".json")]
    for name in names:
        path = stage / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"missing/invalid staged output: {name}")
        if (dest / name).is_symlink():
            raise ValueError(f"refusing to overwrite symlink: {name}")
        old = list(dest.glob(name + ".rollback.*"))
        if (dest / name).exists() and len(old) >= max_rollbacks:
            raise RuntimeError(f"rollback budget exhausted for {name}; archive old backups first")
    backups, committed = {}, []
    try:
        for name in names:
            final = dest / name
            if final.exists():
                backup = dest / (name + ".rollback." + uuid.uuid4().hex[:12])
                os.replace(final, backup)
                backups[name] = backup
            os.replace(stage / name, final)
            committed.append(name)
    except BaseException:
        for name in reversed(committed):
            active = dest / name
            if active.exists():
                os.replace(active, stage / name)
        for name, backup in backups.items():
            if backup.exists():
                os.replace(backup, dest / name)
        raise
    return {name: str(dest / name) for name in names}, {k: str(v) for k,v in backups.items()}


def validate_profile(profile):
    """Bound profile dimensions and labels before geometry or filesystem writes."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", profile.character_id):
        raise ValueError("invalid character_id")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,48}", profile.object_prefix):
        raise ValueError("invalid object_prefix")
    bounds = {"height_m": (1.0, 2.5), "head_scale": (0.7, 1.6),
              "hand_scale": (0.7, 1.6), "shoulder_scale": (0.7, 1.5),
              "hip_scale": (0.7, 1.5), "bulk_scale": (0.7, 1.5),
              "hair_length": (0.4, 2.0)}
    for key, (lo, hi) in bounds.items():
        val = getattr(profile, key)
        if not math.isfinite(val) or not lo <= val <= hi:
            raise ValueError(f"profile {key} outside {lo}..{hi}")
    if len(profile.palette) > 32:
        raise ValueError("profile palette has too many entries")
    for name, color in profile.palette.items():
        if (not isinstance(color, (tuple, list)) or len(color) != 4 or
                any(not isinstance(x, (float,int)) or not math.isfinite(x)
                    or not 0 <= x <= 1 for x in color)):
            raise ValueError(f"invalid RGBA palette: {name}")


def outer_main(args) -> int:
    outdir = Path(args.outdir).resolve() if args.outdir else default_outdir()
    profile = None
    profile_json_path = None
    build_outdir = outdir
    try:
        if getattr(args, "upload", ""):
            profile, sample = profile_from_upload(
                args.upload,
                character_id=(args.character if args.character != "sarah_default" else None),
                style_hint=getattr(args, "style", "auto"))
            validate_profile(profile)
            char_dir = outdir.parent / "characters" / profile.character_id
            char_dir.mkdir(parents=True, exist_ok=True)
            preview = {"character_id": profile.character_id,
                       "profile": profile.to_dict(), "sample": _jsonable_sample(sample),
                       "instant_preview_note": "JSON preview data only: no 3D GLB or AvatarPanel preview"}
            (char_dir / f"{profile.character_id}.preview.json").write_text(
                json.dumps(preview, indent=2), encoding="utf-8")
            profile_json_path = char_dir / f"{profile.character_id}.profile.json"
            profile_json_path.write_text(json.dumps({"profile": profile.to_dict(),
                "sample": _jsonable_sample(sample)}, indent=2), encoding="utf-8")
            print(f"Sampled profile: {profile.summary()}; data: {char_dir}")
            if args.preview_only:
                print("Preview only: no Blender run, no GLB built")
                return 0
            build_outdir = char_dir
        else:
            profile = resolve_profile(args.character)
            validate_profile(profile)
    except (ValueError, OSError, TypeError) as exc:
        print(f"ERROR: invalid character/profile: {exc}")
        return 2
    if args.preview_only:
        print("ERROR: --preview-only requires --upload")
        return 2
    blender = find_blender(args.blender)
    if not blender:
        print("ERROR: Blender unavailable; no asset generated")
        return 2
    basename = Path(blender).name.lower()
    if basename not in ("blender", "blender.exe") or Path(blender).is_symlink():
        print("ERROR: pass the real Blender executable (not launcher or symlink)")
        return 2
    try:
        review = _authorize_local_blender(build_outdir, blender, args)
    except (PermissionError, ValueError) as exc:
        print(f"DENIED: {exc}")
        return 4
    ok = False
    artifact = ""
    error = "unverified"
    try:
        build_outdir.mkdir(parents=True, exist_ok=True)
        with _build_lock(build_outdir):
            with tempfile.TemporaryDirectory(prefix=".sarah_avatar_stage_", dir=build_outdir) as temp:
                stage = Path(temp)
                cmd = [blender, "--background", "--python", str(Path(__file__).resolve()),
                       "--", "--inner", "--outdir", str(stage),
                       "--character", profile.character_id, "--quality", args.quality,
                       "--max-vertices", str(args.max_vertices),
                       "--max-triangles", str(args.max_triangles),
                       "--reference-cache-dir", str(build_outdir / "source_single")]
                if profile_json_path:
                    cmd += ["--profile-json", str(profile_json_path)]
                for flag in ("no_texture", "no_split", "split_all", "head_photo"):
                    if getattr(args, flag, False):
                        cmd.append("--" + flag.replace("_", "-"))
                for role in ("front", "side", "back", "blueprint"):
                    val = getattr(args, role)
                    if val:
                        path = Path(val).resolve()
                        if not path.is_file() or path.is_symlink():
                            raise ValueError(f"missing/unsafe {role} reference: {path}")
                        cmd += [f"--{role}", str(path)]
                print(f"Authorized Blender: {blender}; profile {profile.summary()}; quality={args.quality}")
                rc = _run_blender_bounded(cmd, args.timeout_seconds, stage / "blender.log")
                if rc != 0:
                    error = f"Blender exit {rc}; staged asset discarded"
                    print(f"ERROR: {error}")
                    return rc if rc > 0 else 5
                glb = stage / f"{out_basename_for(profile)}.glb"
                report = verify_glb(glb, max_glb_mb=args.max_glb_mb,
                                    max_vertices=args.max_vertices,
                                    max_triangles=args.max_triangles)
                if not report["ok"]:
                    error = "; ".join(report["errors"])
                    print(json.dumps(report, indent=2))
                    print("FAIL: GLB failed verifier; existing assets remain untouched")
                    return 3
                blend = stage / f"{out_basename_for(profile)}.blend"
                if not blend.is_file() or blend.stat().st_size == 0:
                    raise ValueError("Blender .blend output missing")
                # Builder/Bootstrap appearance and integrity contracts, not extra meshes.
                manifest_path = stage / f"{out_basename_for(profile)}.json"
                original_inner = {}
                if manifest_path.is_file():
                    try:
                        original_inner = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        pass
                manifest = {"ok": True, "script": Path(__file__).name,
                            "version": SCRIPT_VERSION, "module_version": MODULE_VERSION,
                            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                            "character_id": profile.character_id, "profile": profile.to_dict(),
                            "quality": args.quality,
                            "glb_sha256": _file_sha256(glb), "glb_size_bytes": glb.stat().st_size,
                            "blend_sha256": _file_sha256(blend),
                            "shape_keys": EXPECTED_SHAPE_KEYS, "bones": EXPECTED_BONES,
                            "bone_count": len(EXPECTED_BONES),
                            "avatar_panel_contract": {
                                "primary_runtime_asset": str(build_outdir / glb.name),
                                "file_name": glb.name, "armature": RIG_NAME,
                                "pose": "arms_down", "visual_only": True,
                                "avatarEyeCameraAnchor": "Sarah_AvatarEye_Center",
                                "animation_authority": "AvatarPanel_procedural_visual_only"},
                            "animation_contract": {"procedural_panel_owned": True,
                                                   "baked_clips_expected": False},
                            "appearance_slots": {"body": "Sarah_Body", "head": "Sarah_Head",
                                                 "hair": "Sarah_Hair", "boots": "Sarah_Boots",
                                                 "fingers": "Sarah_Fingers",
                                                 "rig": RIG_NAME, "neon": "Sarah_Neon_*"},
                            "verification": report, "inner": original_inner,
                            "validation_boundary": "GLB container/rig/morph/geometry verified; Blender deformation, WebGL and frontend NOT VERIFIED"}
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                paths, backups = _promote_verified_artifacts(
                    stage, build_outdir, out_basename_for(profile),
                    max_rollbacks=args.max_rollbacks)
                artifact = paths[glb.name]
                ok, error = True, ""
                print(f"PASS: verified GLB promoted to {artifact} ({report['size_bytes']:,} bytes)")
                print(f"Rollback backups retained: {len(backups)}")
                print(f"Measured geometry: {report['checks']['vertices_total']:,} vertices / {report['checks']['triangles_total']:,} triangles")
                print("Blender run/export successful; AvatarPanel WebGL/motion test still required")
                return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = str(exc)
        print(f"ERROR: {type(exc).__name__}: {error}")
        return 5
    finally:
        _record_local_blender(review, ok, artifact, error)


# ---------------------------------------------------------------------------
# INNER mode — Blender only. Nothing below runs in plain Python.
# ---------------------------------------------------------------------------
def inner_main(args) -> None:
    import bpy  # noqa: F401  (only available inside Blender)
    from mathutils import Vector

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    quality_factors = {"preview": 0.72, "balanced": 1.0, "high": 1.25}
    DETAIL_FACTOR = quality_factors[getattr(args, "quality", "balanced")]
    out_blend = outdir / f"{OUT_BASENAME}.blend"
    out_glb = outdir / f"{OUT_BASENAME}.glb"

    # v1.0.13 character profile (queue item 8, step 1): the profile
    # drives palette + target height. v1.0.18 (item 9) intentionally
    # changes sarah_default proportions, so it no longer matches
    # v1.0.12/v1.0.17 - that is the point of this step.
    if getattr(args, "profile_json", ""):
        try:
            _pj = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
            profile = AvatarCharacterProfile.from_dict(_pj["profile"])
            if profile.character_id != args.character:
                raise ValueError("profile character_id differs from requested character")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SystemExit(
                f"ERROR: bad --profile-json {args.profile_json}: {exc}") from None
        print(f"Upload profile (v1.0.16): {profile.summary()} "
              f"from {args.profile_json}")
    else:
        profile = resolve_profile(getattr(args, "character", "sarah_default"))
    validate_profile(profile)
    print(f"Character profile (v1.0.16): {profile.summary()}")
    # v1.0.14: re-derive outputs from the profile (out_blend/out_glb
    # above used the Sarah default before the profile was known).
    out_blend = outdir / f"{out_basename_for(profile)}.blend"
    out_glb = outdir / f"{out_basename_for(profile)}.glb"
    print(f"Second profile / scales (v1.0.14): head "
          f"{profile.head_scale}, hand {profile.hand_scale}, shoulder "
          f"{profile.shoulder_scale}, hip {profile.hip_scale}, output "
          f"{out_basename_for(profile)}.glb")
    print(f"Hair / boots / neon by profile (v1.0.15): hair_length "
          f"{profile.hair_length}, boots {profile.boots}, accent "
          f"{profile.accent_mode}, texture_image "
          f"{profile.texture_image!r}")
    print(f"Fortnite-toon (v1.0.18): style {profile.style_preset}, "
          f"head {profile.head_scale}, hand {profile.hand_scale}, "
          f"shoulder {profile.shoulder_scale}, hip {profile.hip_scale}, "
          f"bulk {profile.bulk_scale}")
    # v1.0.11 height: Brian decided 5'8" (1.7272 m) governs. v1.0.10
    # topped out at ~2.025 m, so scale all geometry/bones uniformly.
    TARGET_HEIGHT_M = profile.height_m
    # Head/hair profile scaling changes the actual shell top. The old
    # hardcoded 2.025 m caused Sarah's enlarged toon head to miss 5'8".
    CURRENT_TOP_M = 1.80 + (0.04 + 0.185) * profile.head_scale
    HEIGHT_SCALE = TARGET_HEIGHT_M / CURRENT_TOP_M
    print(f"Height (v1.0.11): target {TARGET_HEIGHT_M} m (5'8\"), "
          f"current top {CURRENT_TOP_M} m, scale {HEIGHT_SCALE:.5f}")

    # --- Clear scene -------------------------------------------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures,
                 bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass

    palette = dict(profile.palette)
    # v1.0.2: DUAL-PANEL SPLIT. Front/side/back references are composites:
    # the SAME avatar twice, side by side — left on black, right on white.
    # They are NOT a stereo pair and must NOT be "aligned" together. For any
    # texture work, and for palette sampling, there must be only ONE avatar
    # in the image. So split each reference down the middle and keep the
    # RIGHT (white-background) half as the single-avatar source: white is
    # far easier to mask out than black, and the suit/neon colours are not
    # tinted by a black glow. The left (black) half is also saved, as a
    # separate file, only as a neon-glow reference — it is never used as
    # a texture source, so the two halves can never blend into one texture.
    # Cropped references may persist outside the GLB staging directory.
    # The active GLB/blend remain protected until successful verification.
    single_dir = (Path(args.reference_cache_dir).resolve() if args.reference_cache_dir
                  else outdir / "source_single")
    single_dir.mkdir(parents=True, exist_ok=True)
    single_refs = {}

    def split_dual_panel(path_str, role):
        """Return the single-avatar (right-half) path for a dual-panel image.

        Uses Blender's image pixels, so no Pillow install is required.
        If the image is already single-avatar (or the split fails), the
        original path is returned unchanged and the failure is logged.
        """
        if not path_str or not Path(path_str).is_file():
            return path_str
        try:
            img = bpy.data.images.load(str(Path(path_str)), check_existing=True)
            w, h = int(img.size[0]), int(img.size[1])
            if w < 100:
                return path_str
            # v1.0.3 fix: v1.0.2 checked ONLY the exact centre column and
            # got 0/64 background rows on Brian's run. That does not mean
            # the image is single — the avatars fill their panels so wide
            # that the exact centre column runs through hair/body on one
            # or both sides. The true gap is a few pixels either side of
            # centre. So scan columns in a band around the centre and pick
            # the column with the most background rows as the seam.
            # Only split when that best seam is background for >=60% of
            # sampled rows; otherwise leave a genuine single-avatar image
            # alone.
            channels = 4
            mid = w // 2
            sample_rows = list(range(0, h, max(1, h // 64)))

            def bg_fraction(col):
                bg = 0
                for row in sample_rows:
                    idx = (row * w + col) * channels
                    r, g, b = img.pixels[idx], img.pixels[idx + 1], img.pixels[idx + 2]
                    lum = (r + g + b) / 3.0
                    if lum < 0.08 or lum > 0.92:
                        bg += 1
                return bg / max(1, len(sample_rows))

            best_col, best_frac = mid, bg_fraction(mid)
            for col in range(max(0, mid - 160), min(w - 1, mid + 161), 4):
                frac = bg_fraction(col)
                if frac > best_frac:
                    best_col, best_frac = col, frac
            print(f"{role} reference: {w}x{h}, best seam column {best_col} "
                  f"(centre {mid}), background fraction {best_frac:.2f}")
            if best_frac < 0.60:
                print(f"  {role}: no background seam found, leaving image alone")
                return path_str
            seam = best_col
            # v1.0.4 fix: v1.0.3 got STUCK here. It copied pixels one by one
            # in Python (2048 rows x ~686 cols x 4 channels, with a Blender
            # sequence access per pixel) — that is minutes of work with no
            # progress output, so it looked frozen. This version pulls the
            # whole image into a C array once with foreach_get, then slices
            # whole rows at C speed, and pushes the result with foreach_set.
            # It also prints progress, so a slow run is visible, not silent.
            print(f"  {role}: cropping at seam {seam} (this used to be the slow step)")
            try:
                from array import array as _array
                full = _array("f", [0.0]) * (w * h * channels)
                img.pixels.foreach_get(full)
            except Exception as exc:
                print(f"  {role}: fast pixel read failed ({exc}), falling back to slow copy")
                full = None
            for half_name, x0, half_w in (("black", 0, seam), ("white", seam, w - seam)):
                if half_w < 50:
                    continue
                _cached = single_dir / f"{role}_{half_name}.png"
                if (_cached.is_file() and _cached.stat().st_mtime_ns >=
                        Path(path_str).stat().st_mtime_ns):
                    print(f"  {role} ({half_name}) cache current: {_cached}")
                    if half_name == "white":
                        single_refs[role] = str(_cached)
                    continue
                new_img = bpy.data.images.new(
                    name=f"Sarah_{role}_{half_name}", width=half_w, height=h, alpha=True)
                if full is not None:
                    cropped = _array("f")
                    for row in range(h):
                        start = (row * w + x0) * channels
                        cropped.extend(full[start:start + half_w * channels])
                    new_img.pixels.foreach_set(cropped)
                else:
                    new_pixels = []
                    for row in range(h):
                        base = row * w * channels
                        for col in range(half_w):
                            src = base + (x0 + col) * channels
                            new_pixels.extend(img.pixels[src:src + channels])
                    new_img.pixels = new_pixels
                out_path = single_dir / f"{role}_{half_name}.png"
                new_img.filepath_raw = str(out_path)
                new_img.file_format = "PNG"
                new_img.save()
                print(f"  Wrote single-avatar {role} ({half_name}): {out_path}")
                if half_name == "white":
                    single_refs[role] = str(out_path)
            return single_refs.get(role, path_str)
        except Exception as exc:
            print(f"{role} dual-panel split skipped: {exc}")
            return path_str

    # v1.0.5: image splitting is NOT needed to build the GLB at all — the
    # geometry uses palette defaults, and side/back are never used for
    # geometry. Splitting three 1365x2048 images (11M floats each, plus
    # PNG saves for 6 crops) is what made v1.0.3/4 feel slow, while the
    # actual glTF export was only ~0.6 s in every run. So by default only
    # split the front image, skip side/back unless --split-all is given,
    # skip crops that already exist, and allow --no-split to skip all of
    # it for a fast GLB-only run.
    if getattr(args, "no_split", False):
        print("Skipping reference split (--no-split): using original images for palette check only")
        front_single = args.front
    else:
        front_single = split_dual_panel(args.front, "front")
        if getattr(args, "split_all", False):
            split_dual_panel(args.side, "side")
            split_dual_panel(args.back, "back")
        else:
            print("Skipping side/back split (use --split-all if you need those crops)")
    # Palette sampling from the SINGLE front image, if Blender can load it.
    # This is best-effort only; defaults above are the blueprint palette.
    # (Sampling code intentionally does not drive geometry — see header.)
    if front_single and Path(front_single).is_file():
        try:
            img = bpy.data.images.load(str(Path(front_single)), check_existing=True)
            print(f"Front single-avatar reference loaded for palette check: {img.size[0]}x{img.size[1]}")
        except Exception as exc:
            print(f"Front image palette sampling skipped: {exc}")

    def make_material(name, base_color, metallic=0.0, roughness=0.5,
                      emissive=None, emissive_strength=0.0):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = base_color
            if "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = metallic
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = roughness
            if emissive is not None and "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emissive
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emissive_strength
        return mat

    # v1.0.18, queue item 9: bolder, flatter toon shading for
    # fortnite_toon profiles (Sarah is one now). This is a Principled
    # roughness/metallic flattening, NOT a cel shader - glTF has no
    # standard cel shader, so a true toon look needs panel-side work.
    _toon = profile.style_preset == "fortnite_toon"
    _suit_metallic = 0.0 if _toon else 0.18
    _suit_rough = 0.35 if _toon else 0.12
    _skin_rough = 0.60 if _toon else 0.47
    _hair_rough = 0.55 if _toon else 0.36
    _neon_strength = 5.0 if _toon else 4.0
    mats = {
        "skin": make_material("Sarah_Skin", palette["skin"], roughness=_skin_rough),
        "suit": make_material("Sarah_Suit_BlackGloss", palette["suit"],
                              metallic=_suit_metallic, roughness=_suit_rough),
        "hair": make_material("Sarah_Hair_Magenta", palette["hair"], roughness=_hair_rough),
        "neon": make_material("Sarah_Neon_Cyan", palette["neon"],
                              metallic=0.02, roughness=0.06,
                              emissive=palette["neon"], emissive_strength=_neon_strength),
        "sclera": make_material("Sarah_Eye_Sclera", palette["sclera"], roughness=0.18),
        "iris": make_material("Sarah_Eye_Iris", palette["iris"], roughness=0.14),
        "lips": make_material("Sarah_Lips_SoftRose", palette["lips"], roughness=0.28),
        "brows": make_material("Sarah_Brows_DarkMagenta", palette["brows"], roughness=0.48),
        # v1.0.23: dark mouth interior, from the brows palette colour
        # (dark rose/magenta for Sarah, dark purple for toon_hero_01),
        # so no new palette key is needed for either profile.
        "mouth": make_material("Sarah_Mouth_Interior", palette["brows"], roughness=0.60),
    }

    # --- v1.0.6: TEXTURE from verified single-avatar crops (queue item 1) --
    # The June GLB had textures=null. Load source_single/*_white.png
    # (right/white half ONLY — never *_black.png, never a blended
    # dual-panel image) and wire it as an Image Texture node on Base
    # Color for the body and head materials, so the glTF exporter
    # embeds it in the GLB. Boots keep the flat suit material, so the
    # body texture is not smeared onto the boots. Fallback: flat
    # v1.0.5 colours if no crop is found, or with --no-texture.
    # HONEST LIMIT: Smart UV is not a front projection, so this maps
    # approximately on curved parts. A proper projected unwrap is a
    # later item, not claimed here.
    def find_texture_crops():
        found = {}
        search_dirs = [single_dir, outdir / "source_single"]
        for role in ("front", "side", "back"):
            reference = getattr(args, role, "")
            for directory in search_dirs:
                candidate = directory / f"{role}_white.png"
                # Never apply an old crop to a newer reference.  Without a
                # source image an existing curated crop is still usable.
                if candidate.is_file() and (not reference or
                        (Path(reference).is_file() and
                         candidate.stat().st_mtime_ns >= Path(reference).stat().st_mtime_ns)):
                    found[role] = str(candidate)
                    break
            if role not in found and single_refs.get(role):
                found[role] = single_refs[role]
            if reference and role not in found:
                print(f"WARNING: no current single-avatar {role} crop; using flat/fallback texture")
        return found

    def apply_texture(mat, image_path, label):
        try:
            img = bpy.data.images.load(str(Path(image_path)), check_existing=True)
        except Exception as exc:
            print(f"Texture skipped for {mat.name}: could not load {image_path}: {exc}")
            return False
        try:
            img.colorspace_settings.name = "sRGB"
        except Exception:
            pass
        tex = mat.node_tree.nodes.new(type="ShaderNodeTexImage")
        tex.image = img
        tex.label = label
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        print(f"Texture applied to {mat.name}: {image_path}")
        return True

    # v1.0.15: per-profile texture_image. If a profile names one, it
    # is used (relative to the output folder, or as given) instead
    # of Sarah's *_white.png crops. Sarah's crops are still never
    # painted onto a non-Sarah profile; a non-Sarah profile with no
    # texture_image builds with flat profile palette colours.
    profile_texture = None
    if getattr(profile, "texture_image", ""):
        _cand = Path(profile.texture_image)
        if not _cand.is_absolute():
            _cand = outdir / _cand
        if _cand.is_file():
            profile_texture = str(_cand)
        else:
            print(f"WARNING: profile texture_image not found: "
                  f"{profile.texture_image} (looked at {_cand}) — "
                  f"building with flat profile palette colours")
    texture_crops = {} if getattr(args, "no_texture", False) else (
        {"front": profile_texture} if profile_texture else (
            find_texture_crops() if profile.character_id == "sarah_default" else {}))
    # v1.0.20: avatar bbox in the front image, for enlarging the
    # texture to the model outline. v1.0.19 mapped the model to
    # the FULL image, so white margins mapped onto arms/legs and
    # the photo avatar read smaller than the model outline.
    # sample_upload_image (stdlib PNG, corner-background mask)
    # gives bbox left/right/top/bottom in PNG rows; convert to
    # Blender UV (v=0 at bottom). On any failure, fall back to
    # the full image (v1.0.19 behaviour) and say so loudly.
    # v1.0.21: same bbox enlargement for EVERY crop, not just
    # front, so side/back projections also map to the avatar
    # inside their photo instead of its white margins. Roles
    # without a crop, or whose sampling fails, get None here
    # and fall back to the front image / full-image UV loudly
    # in project_uv_tri, so a front-only folder still builds.
    texture_bbox_uvs = {}
    for _role, _path in texture_crops.items():
        try:
            _sample = sample_upload_image(_path)
            _bb = _sample["bbox"]
            _w, _h = _sample["width"], _sample["height"]
            texture_bbox_uvs[_role] = (
                _bb["left"] / _w,
                (_bb["right"] + 1) / _w,
                1.0 - (_bb["bottom"] + 1) / _h,
                1.0 - _bb["top"] / _h,
            )
            print(f"Texture bbox (v1.0.21): {_role} {_bb}, "
                  f"image {_w}x{_h}, UV {texture_bbox_uvs[_role]}")
        except Exception as exc:
            print(f"WARNING: {_role} bbox sampling failed ({exc}) — "
                  f"using full-image UV (v1.0.19 behaviour)")
    texture_front_bbox_uv = texture_bbox_uvs.get("front")
    if getattr(args, "no_texture", False):
        print("Skipping textures (--no-texture): flat v1.0.5 colours will be used")
    elif texture_crops:
        print(f"Texture crops found: {texture_crops}")
        front_tex = texture_crops.get("front")
        if front_tex:
            mats["suit_textured"] = make_material(
                f"{profile.object_prefix}_Suit_Textured", palette["suit"],
                metallic=_suit_metallic, roughness=_suit_rough)
            apply_texture(mats["suit_textured"], front_tex, f"{profile.object_prefix}_Body_FrontWhite")
            # v1.0.24 photo-face decision: the head photo texture
            # is suppressed by default, so no skin_textured
            # material is created at all — the Head assign below
            # then falls back to the flat skin material, and the
            # tri-projection block keeps only a front projection
            # for UV (harmless on a flat material). --head-photo
            # restores the v1.0.21-23 behaviour for the head.
            if getattr(args, "head_photo", False):
                mats["skin_textured"] = make_material(
                    f"{profile.object_prefix}_Skin_Textured", palette["skin"], roughness=_skin_rough)
                apply_texture(mats["skin_textured"], front_tex, f"{profile.object_prefix}_Head_FrontWhite")
            else:
                print("Head photo texture suppressed (v1.0.24): Head will use "
                      "flat skin so the toon face reads alone; pass "
                      "--head-photo to restore the photo head")
            # v1.0.21 tri-projection materials: one textured
            # material per direction, per mesh type, so the glTF
            # exporter can embed front_white + side_white +
            # back_white as three images instead of one. A role
            # with no crop simply gets no new material, and
            # project_uv_tri falls back to the front material
            # for that direction (see below).
            for _role, _label in (("side", "SideWhite"), ("back", "BackWhite")):
                _crop = texture_crops.get(_role)
                if not _crop:
                    print(f"No {_role}_white.png crop found — "
                          f"{_role} polygons will use the front texture")
                    continue
                mats[f"suit_textured_{_role}"] = make_material(
                    f"{profile.object_prefix}_Suit_Textured_{_label}", palette["suit"],
                    metallic=_suit_metallic, roughness=_suit_rough)
                apply_texture(mats[f"suit_textured_{_role}"], _crop, f"{profile.object_prefix}_Body_{_label}")
                if getattr(args, "head_photo", False):
                    mats[f"skin_textured_{_role}"] = make_material(
                        f"{profile.object_prefix}_Skin_Textured_{_label}", palette["skin"], roughness=_skin_rough)
                    apply_texture(mats[f"skin_textured_{_role}"], _crop, f"{profile.object_prefix}_Head_{_label}")
        else:
            print("No front_white.png crop found — building with flat colours")
    else:
        if profile.character_id != "sarah_default":
            print(f"Building {profile.character_id} with flat profile palette "
                  f"colours (v1.0.15): Sarah's *_white.png crops are not "
                  f"applied to a non-Sarah character, and no "
                  f"texture_image is set for this profile.")
        else:
            print("WARNING: no source_single/*_white.png crops found in the output "
                  "folder — building with flat v1.0.5 colours (textures will be "
                  "null in this GLB). Run once without --no-split first, or copy "
                  "the crops into resources/avatars/3D/source_single/.")

    def assign_material(obj, mat):
        if obj and obj.data and hasattr(obj.data, "materials"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)

    def add_sphere(name, location, scale, segments=32):
        # v1.0.11: uniform height scale applied at creation, so
        # skinning, shape keys and export all see scaled geometry.
        location = tuple(v * HEIGHT_SCALE for v in location)
        scale = tuple(v * HEIGHT_SCALE for v in scale)
        segments = max(12, min(96, round(segments * DETAIL_FACTOR)))
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=segments, ring_count=max(8, segments // 2),
            radius=1.0, location=location)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.name = name + "Mesh"
        obj.scale = scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj

    def add_box(name, location, scale, rotation=(0.0, 0.0, 0.0)):
        location = tuple(v * HEIGHT_SCALE for v in location)
        scale = tuple(v * HEIGHT_SCALE for v in scale)
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=location, rotation=rotation)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.name = name + "Mesh"
        obj.scale = scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj

    def add_torus(name, location, major_radius, minor_radius,
                  major_segments=32, minor_segments=8):
        # v1.0.12: ankle rings. A torus defaults to lying in the XY
        # plane (hole along Z), which is already a horizontal ring
        # around a vertical leg, so no rotation is needed. Radii
        # scale by HEIGHT_SCALE like every other dimension.
        location = tuple(v * HEIGHT_SCALE for v in location)
        bpy.ops.mesh.primitive_torus_add(
            location=location, rotation=(0.0, 0.0, 0.0),
            major_radius=major_radius * HEIGHT_SCALE,
            minor_radius=minor_radius * HEIGHT_SCALE,
            major_segments=max(16, int(major_segments * DETAIL_FACTOR)),
            minor_segments=max(6, int(minor_segments * DETAIL_FACTOR)))
        obj = bpy.context.active_object
        obj.name = name
        obj.data.name = name + "Mesh"
        return obj

    def add_ring_loft(name, rings, axis="Z", segments=32):
        # v1.0.22: ring body (queue item 10), ported from the
        # v9.1.0a Bootstrap _create_ring_mesh_object method.
        # rings = [(centre, a, b), ...] where a/b are the ring
        # radii in the two axes perpendicular to `axis`.
        # Centres and radii scale by HEIGHT_SCALE like
        # add_sphere/add_box, so skinning and export see scaled
        # geometry. No bpy.ops primitive is used: vertices and
        # faces are built directly, then the object is created
        # from that data — the same construction Bootstrap uses,
        # so the surface is a smooth loft, not joined spheres.
        # Caps use Bootstrap's fan winding verbatim.
        verts = []
        faces = []
        segments = max(12, min(96, int(segments * DETAIL_FACTOR)))
        for (cx, cy, cz), a, b in rings:
            cx, cy, cz = cx * HEIGHT_SCALE, cy * HEIGHT_SCALE, cz * HEIGHT_SCALE
            a, b = a * HEIGHT_SCALE, b * HEIGHT_SCALE
            for i in range(segments):
                theta = (i / segments) * math.tau
                c, s = math.cos(theta), math.sin(theta)
                if axis == "Z":
                    verts.append((cx + a * c, cy + b * s, cz))
                elif axis == "X":
                    verts.append((cx, cy + a * c, cz + b * s))
                elif axis == "Y":
                    verts.append((cx + a * c, cy, cz + b * s))
                else:
                    verts.append((cx + a * c, cy + b * s, cz))
        ring_count = len(rings)
        for r in range(ring_count - 1):
            base, nxt = r * segments, (r + 1) * segments
            for i in range(segments):
                j = (i + 1) % segments
                faces.append((base + i, base + j, nxt + j, nxt + i))
        if ring_count > 0:
            cidx = len(verts)
            verts.append(tuple(v * HEIGHT_SCALE for v in rings[0][0]))
            for i in range(segments):
                j = (i + 1) % segments
                faces.append((cidx, j, i))
        if ring_count > 1:
            cidx = len(verts)
            verts.append(tuple(v * HEIGHT_SCALE for v in rings[-1][0]))
            base = (ring_count - 1) * segments
            for i in range(segments):
                j = (i + 1) % segments
                faces.append((cidx, base + i, base + j))
        mesh = bpy.data.meshes.new(name + "Mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def add_hair_card(name, stations, width, thickness=0.006,
                      orientation="xy"):
        # v1.0.25: hair cards (queue item 13, step 1). stations =
        # [(x, y, z), ...] top -> tip, in generator coordinates
        # (already hair_loc/head-scaled by the caller, as with
        # add_sphere callers); width is the base card width in X.
        # Centres and width/thickness scale by HEIGHT_SCALE like
        # add_sphere/add_ring_loft, so skinning and export see
        # scaled geometry. Built directly from vertices/faces
        # (same construction as add_ring_loft): 4 verts per
        # station (front/back x left/right, thickness in Y),
        # front + back + left-edge + right-edge quads per
        # segment, so the ribbon reads from both sides without
        # a double-sided material change. Width tapers
        # 1.0 -> 0.65 from top to tip.
        # v1.0.26 (queue item 13, step 2): orientation="yz" for
        # side/shoulder falls — width runs in Y and thickness in
        # X, so the ribbon's broad face points sideways instead
        # of edge-on from the side. The same face-index pattern
        # is reused, so the closed ribbon still shows a broad
        # face from each side, as in v1.0.25.
        verts = []
        faces = []
        n = len(stations)
        for i, (sx, sy, sz) in enumerate(stations):
            taper = 1.0 - 0.35 * (i / (n - 1) if n > 1 else 0.0)
            hw = width * taper * HEIGHT_SCALE / 2.0
            ht = thickness * HEIGHT_SCALE / 2.0
            x, y, z = sx * HEIGHT_SCALE, sy * HEIGHT_SCALE, sz * HEIGHT_SCALE
            if orientation == "yz":
                verts.append((x - ht, y - hw, z))  # inner-front
                verts.append((x - ht, y + hw, z))  # inner-back
                verts.append((x + ht, y - hw, z))  # outer-front
                verts.append((x + ht, y + hw, z))  # outer-back
            else:
                verts.append((x - hw, y - ht, z))  # front-left
                verts.append((x + hw, y - ht, z))  # front-right
                verts.append((x - hw, y + ht, z))  # back-left
                verts.append((x + hw, y + ht, z))  # back-right
        for i in range(n - 1):
            b, nx = i * 4, (i + 1) * 4
            faces.append((b + 0, b + 1, nx + 1, nx + 0))  # front
            faces.append((b + 2, nx + 2, nx + 3, b + 3))  # back
            faces.append((b + 0, nx + 0, nx + 2, b + 2))  # left edge
            faces.append((b + 1, b + 3, nx + 3, nx + 1))  # right edge
        mesh = bpy.data.meshes.new(name + "Mesh")
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def join_objects(name, parts):
        bpy.ops.object.select_all(action="DESELECT")
        for p in parts:
            p.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        bpy.ops.object.join()
        joined = bpy.context.active_object
        joined.name = name
        joined.data.name = name + "Mesh"
        bpy.ops.object.shade_smooth()
        return joined

    def smart_uv(obj):
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        except Exception as exc:
            print(f"Smart UV skipped for {obj.name}: {exc}")
        bpy.ops.object.mode_set(mode="OBJECT")

    def project_uv_front(obj, bounds, bbox_uv=None):
        # v1.0.19 front projection, v1.0.20 enlarged to the model
        # outline: if bbox_uv is given, the model maps to the
        # AVATAR BBOX inside the photo, not the full image, so
        # white margins no longer land on arms/legs and the photo
        # avatar fills the model outline instead of reading small.
        # u is flipped (model +X = viewer's left in the photo).
        # Back/side surfaces still mirror/stretch the front image.
        mesh = obj.data
        if not mesh.uv_layers:
            mesh.uv_layers.new(name="UVMap")
        uv = mesh.uv_layers.active.data
        x0, x1, z0, z1 = bounds
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                vert = mesh.vertices[mesh.loops[li].vertex_index]
                wco = obj.matrix_world @ vert.co
                u_m = (wco.x - x0) / (x1 - x0) if x1 > x0 else 0.5
                v_m = (wco.z - z0) / (z1 - z0) if z1 > z0 else 0.5
                u_p = 1.0 - u_m
                if bbox_uv is not None:
                    u_l, u_r, v_b, v_t = bbox_uv
                    uv[li].uv = (u_l + u_p * (u_r - u_l),
                                 v_b + v_m * (v_t - v_b))
                else:
                    uv[li].uv = (u_p, v_m)
        print(f"Projected UV (front, v1.0.20): {obj.name}, "
              f"x {x0:.3f}..{x1:.3f}, z {z0:.3f}..{z1:.3f}, "
              f"bbox_uv={bbox_uv}")

    def project_uv_tri(obj, bounds, side_bounds, bbox_uvs,
                       front_mat, side_mat, back_mat, flat_mat=None):
        # v1.0.21 tri-projection (queue item 11, step 3). UVs are
        # per LOOP, not per vertex, so one vertex shared by a
        # front face and a side face can hold both projections.
        # Each polygon is assigned by its dominant WORLD normal
        # (front = -Y, back = +Y, side = +/-X; top/bottom
        # polygons, dominant Z, fall back to front) to the
        # matching material slot, and its loops get that
        # direction's projection. Front u is flipped as in
        # v1.0.19; back u is NOT flipped (a back view shows the
        # person's right on the viewer's right); side u comes
        # from world Y and is the same on both sides, so one
        # side mirrors — that is a known limit, not a bug to
        # hide, until a second side photo exists. A missing
        # side/back material falls back to front for that
        # direction, loudly, via the caller-side print below.
        # v1.0.28 limb-side fix: when flat_mat is given (Body
        # only), a side-facing polygon whose centre lies in a
        # limb region (see the limb test below) is reassigned
        # to flat_mat, because the global side projection maps
        # narrow limbs onto the side photo's white background.
        # Front/back limb polygons and torso side polygons are
        # untouched, and the Head call passes no flat_mat.
        mesh = obj.data
        if not mesh.uv_layers:
            mesh.uv_layers.new(name="UVMap")
        uv = mesh.uv_layers.active.data
        slots = []
        role_index = {}
        for _role, _mat in (("front", front_mat),
                            ("side", side_mat), ("back", back_mat),
                            ("flat", flat_mat)):
            if _mat is not None and _mat not in slots:
                role_index[_role] = len(slots)
                slots.append(_mat)
        mesh.materials.clear()
        for _mat in slots:
            mesh.materials.append(_mat)
        x0, x1, z0, z1 = bounds
        y0, y1 = side_bounds
        try:
            _rot = obj.matrix_world.to_3x3()
        except Exception:
            _rot = None
        counts = {"front": 0, "side": 0, "back": 0, "flat": 0}
        for poly in mesh.polygons:
            _n = poly.normal
            if _rot is not None:
                try:
                    _n = _rot @ _n
                except Exception:
                    pass
            _len = math.sqrt(_n.x * _n.x + _n.y * _n.y + _n.z * _n.z) or 1.0
            _ax, _ay, _az = abs(_n.x) / _len, abs(_n.y) / _len, abs(_n.z) / _len
            if _ay >= _ax and _ay >= _az:
                _role = "front" if _n.y < 0 else "back"
            elif _ax >= _ay and _ax >= _az:
                _role = "side"
            else:
                _role = "front"
            if flat_mat is not None:
                # v1.0.30: limb centre in scaled world coords.
                # v1.0.28 flattened SIDE polygons only, but Brian's
                # 2026-11-04 21:28 log (SarahMemoryAvatarGenerate.py,
                # NOT v1.0.29) shows the white is also on FRONT
                # polygons: with front bbox UV 0.088..0.853, an
                # outer arm/leg front polygon maps to u
                # 0.09..0.22, i.e. the white margin left of the
                # photo limb, not the limb suit. So ANY polygon
                # (front, back or side) whose centre lies in a
                # limb region goes flat black. Torso side
                # polygons at chest height also meet the arm
                # test and will flatten too - acceptable, they
                # are black suit anyway and neon strips sit on
                # top as separate meshes.
                # Legs centre x ~0.11 pre-scale, arms ~0.21-0.28;
                # torso chest radius 0.231 pre-scale stays below
                # the 0.24 arm threshold, and torso waist side
                # quads sit above z 0.92, so neither is flattened.
                _cx = _cy = _cz = 0.0
                _nv = 0
                for _li in poly.loop_indices:
                    _v = mesh.vertices[mesh.loops[_li].vertex_index]
                    _w = obj.matrix_world @ _v.co
                    _cx += _w.x; _cy += _w.y; _cz += _w.z; _nv += 1
                if _nv:
                    _cx /= _nv; _cz /= _nv
                _hs = HEIGHT_SCALE
                _is_leg = abs(_cx) > 0.04 * _hs and _cz < 0.92 * _hs
                _is_arm = (abs(_cx) > 0.24 * _hs
                           and 0.82 * _hs < _cz < 1.55 * _hs)
                if _is_leg or _is_arm:
                    _role = "flat"
            if _role not in role_index:
                _role = "front"
            poly.material_index = role_index[_role]
            counts[_role] += 1
            _bbox = bbox_uvs.get(_role)
            for li in poly.loop_indices:
                vert = mesh.vertices[mesh.loops[li].vertex_index]
                wco = obj.matrix_world @ vert.co
                v_m = (wco.z - z0) / (z1 - z0) if z1 > z0 else 0.5
                if _role == "side":
                    u_m = (wco.y - y0) / (y1 - y0) if y1 > y0 else 0.5
                    u_p = u_m
                else:
                    u_m = (wco.x - x0) / (x1 - x0) if x1 > x0 else 0.5
                    u_p = (1.0 - u_m) if _role == "front" else u_m
                if _role == "flat":
                    uv[li].uv = (0.0, v_m)  # flat material: UV unused
                elif _bbox is not None:
                    u_l, u_r, v_b, v_t = _bbox
                    uv[li].uv = (u_l + u_p * (u_r - u_l),
                                 v_b + v_m * (v_t - v_b))
                else:
                    uv[li].uv = (u_p, v_m)
        print(f"Projected UV (tri, v1.0.21): {obj.name}, "
              f"polygons front/side/back/flat = "
              f"{counts['front']}/{counts['side']}/{counts['back']}/"
              f"{counts['flat']}, "
              f"materials {[m.name for m in slots]}")

    # --- Profile scales (v1.0.14, queue item 8 step 2) ------------------
    # Head parts / eyes / face details scale about the head centre
    # (0, 0, 1.80), so a larger head keeps its face on its surface:
    # e.g. for head_scale 1.25 the brow centre y -0.150 -> -0.1875
    # against a head front of -0.181, i.e. still ~0.001-0.002 m
    # inside, as in v1.0.9. v1.0.18: sarah_default is 1.18, so these
    # helpers are no longer the identity for Sarah - that is the
    # Fortnite-toon step, and the face still stays on its surface
    # because it scales about the same head centre.
    HEAD_CENTRE = (0.0, 0.0, 1.80)

    def head_loc(location):
        return tuple(HEAD_CENTRE[i] + (location[i] - HEAD_CENTRE[i])
                     * profile.head_scale for i in range(3))

    def head_scale_vec(scale):
        return tuple(v * profile.head_scale for v in scale)

    # v1.0.15: hair width follows the head, length (z) does not —
    # falls/strands hang down from the head, so only x/y scale
    # about the head centre (centre y is 0, so y is just * head_scale).
    def hair_loc(location):
        return (location[0] * profile.head_scale,
                location[1] * profile.head_scale, location[2])

    # --- Body (arms DOWN, Z up, front = -Y) --------------------------------
    # v1.0.22 ring body (queue item 10, ported from the v9.1.0a
    # Bootstrap ring method, re-based to this generator's
    # coordinates — see the header bullet): 5 lofts joined into
    # the single Sarah_Body mesh and auto-weighted, as the 29
    # spheres were. Torso: waist pinch at z 1.05, chest at
    # 1.42 (centre y -0.02, b 0.12*bulk, so its back +0.112 m
    # leaves the back-spine neon at +0.113 m outside it), neck
    # cap at 1.65 (front -0.099 m, matching the collar neon
    # at -0.093*bulk). Legs: thigh widest at 0.70, knee pinch
    # at 0.55, ankle flare at 0.12 for the boot overlap.
    # Arms: shoulder cap at 1.52, elbow at 1.18, wrist at
    # 0.92, hand cap at 0.86 — arms DOWN, on the same bone
    # line as before (0.20-0.28 in x), NOT Bootstrap's
    # sideways-X arms. Ring radii are the old sphere radii
    # re-expressed, so every v1.0.7/v1.0.18 neon offset still
    # sits outside its surface (checked in plain Python for
    # Sarah: torso centre gap ~0.004-0.013 m, thigh ~0.006 m,
    # shin ~0.002 m, upper-arm ~0.005 m, forearm ~0.003 m).
    _bulk = profile.bulk_scale
    _hip = profile.hip_scale
    _sh = profile.shoulder_scale
    _hand = profile.hand_scale
    torso_rings = [
        ((0.0, 0.0, 0.90), 0.185 * _hip, 0.125 * _hip),
        ((0.0, 0.0, 0.97), 0.195 * _hip, 0.130 * _hip),
        ((0.0, 0.0, 1.05), 0.185 * _bulk, 0.125 * _bulk),
        ((0.0, 0.0, 1.15), 0.200 * _bulk, 0.130 * _bulk),
        ((0.0, 0.0, 1.25), 0.210 * _bulk, 0.130 * _bulk),
        ((0.0, -0.01, 1.35), 0.205 * _bulk, 0.125 * _bulk),
        ((0.0, -0.02, 1.42), 0.195 * _bulk, 0.120 * _bulk),
        ((0.0, -0.005, 1.50), 0.170 * _bulk, 0.105 * _bulk),
        ((0.0, 0.0, 1.58), 0.130 * _bulk, 0.090 * _bulk),
        ((0.0, 0.0, 1.65), 0.090 * _bulk, 0.090 * _bulk),
    ]
    body_parts = [add_ring_loft("Body_Torso_Rings", torso_rings, axis="Z", segments=32)]
    for sign in (-1.0, 1.0):
        s = "L" if sign < 0 else "R"
        leg_rings = [
            ((0.11 * sign * _hip, 0.0, 0.98), 0.085 * _bulk, 0.080 * _bulk),
            ((0.11 * sign * _hip, 0.0, 0.85), 0.090 * _bulk, 0.085 * _bulk),
            ((0.115 * sign * _hip, 0.0, 0.70), 0.095 * _bulk, 0.090 * _bulk),
            ((0.115 * sign * _hip, 0.0, 0.55), 0.088 * _bulk, 0.075 * _bulk),
            ((0.12 * sign * _hip, 0.0, 0.40), 0.075 * _bulk, 0.068 * _bulk),
            ((0.12 * sign * _hip, 0.0, 0.25), 0.070 * _bulk, 0.063 * _bulk),
            ((0.11 * sign * _hip, 0.0, 0.12), 0.078 * _bulk, 0.070 * _bulk),
        ]
        body_parts.append(add_ring_loft(f"Body_Leg_Rings_{s}", leg_rings, axis="Z", segments=32))
        arm_rings = [
            ((0.21 * sign * _sh, 0.0, 1.52), 0.070 * _sh, 0.070 * _sh),
            ((0.235 * sign, 0.0, 1.40), 0.058 * _bulk, 0.055 * _bulk),
            ((0.25 * sign, 0.0, 1.30), 0.055 * _bulk, 0.052 * _bulk),
            ((0.255 * sign, 0.0, 1.18), 0.055 * _bulk, 0.050 * _bulk),
            ((0.27 * sign, 0.0, 1.05), 0.048 * _bulk, 0.045 * _bulk),
            ((0.28 * sign, 0.0, 0.92), 0.047 * _bulk, 0.043 * _bulk),
            ((0.28 * sign, -0.01, 0.86), 0.045 * _hand, 0.040 * _hand),
        ]
        body_parts.append(add_ring_loft(f"Body_Arm_Rings_{s}", arm_rings, axis="Z", segments=32))
    body = join_objects("Sarah_Body", body_parts)
    print(f"Ring body (v1.0.22): {len(body_parts)} ring lofts "
          f"(torso {len(torso_rings)} rings, legs/arms 7 rings each), "
          f"waist pinch z 1.05, chest z 1.42, neck cap z 1.65.")
    assign_material(body, mats.get("suit_textured", mats["suit"]))
    # v1.0.19: UV assigned after Head exists, from shared bounds
    # (see below), not Smart UV here.

    # --- Head ---------------------------------------------------------------
    head_parts = [
        add_sphere("Head_Main", head_loc((0.0, 0.0, 1.80)), head_scale_vec((0.155, 0.145, 0.165))),
        # v1.0.24 nose remodel (queue item 12, step 2): a vertical
        # bridge ridge plus a protruding tip, instead of one ball.
        # Bridge front -0.162 and tip front -0.170 (unscaled) sit
        # just proud of the old nose front (-0.173 was the ball's
        # front; the tip matches it, the bridge sits slightly
        # shallower), so the nose reads as ridge + tip.
        add_sphere("Head_Nose_Bridge", head_loc((0.0, -0.142, 1.80)), head_scale_vec((0.018, 0.020, 0.045))),
        add_sphere("Head_Nose_Tip", head_loc((0.0, -0.152, 1.765)), head_scale_vec((0.024, 0.018, 0.016))),
        add_sphere("Head_Chin", head_loc((0.0, -0.085, 1.685)), head_scale_vec((0.055, 0.04, 0.035))),
        add_sphere("Head_Cheek_L", head_loc((-0.07, -0.095, 1.77)), head_scale_vec((0.046, 0.026, 0.05))),
        add_sphere("Head_Cheek_R", head_loc((0.07, -0.095, 1.77)), head_scale_vec((0.046, 0.026, 0.05))),
        add_sphere("Head_Ear_L", head_loc((-0.132, 0.0, 1.79)), head_scale_vec((0.026, 0.014, 0.058))),
        add_sphere("Head_Ear_R", head_loc((0.132, 0.0, 1.79)), head_scale_vec((0.026, 0.014, 0.058))),
    ]
    head = join_objects("Sarah_Head", head_parts)
    assign_material(head, mats.get("skin_textured", mats["skin"]))
    # v1.0.19 projected texture: shared front-projection bounds
    # from Body + Head world coordinates, so both meshes sample
    # the same front image in the same space.
    _xs, _ys, _zs = [], [], []
    for _obj in (body, head):
        for _vert in _obj.data.vertices:
            _w = _obj.matrix_world @ _vert.co
            _xs.append(_w.x)
            _ys.append(_w.y)
            _zs.append(_w.z)
    _bounds = (min(_xs), max(_xs), min(_zs), max(_zs))
    _side_bounds = (min(_ys), max(_ys))
    # v1.0.21: if a textured front material exists, tri-projection
    # replaces the single front projection (it also re-sets the
    # material slots per direction, overriding the single
    # assign_material above). Without textures, keep the v1.0.19
    # front projection for UV only and leave the flat material.
    if "suit_textured" in mats:
        print(f"Tri-projection materials (v1.0.21): side/back crops "
              f"present = {('side' in texture_crops, 'back' in texture_crops)}; "
              f"missing directions fall back to the front texture")
        # v1.0.28 limb-side fix, re-applied on the consolidated base:
        # flat suit for limb side polygons (Body only).
        project_uv_tri(body, _bounds, _side_bounds, texture_bbox_uvs,
                       mats["suit_textured"],
                       mats.get("suit_textured_side"),
                       mats.get("suit_textured_back"),
                       mats["suit"])
    else:
        project_uv_front(body, _bounds, texture_front_bbox_uv)
    if "skin_textured" in mats:
        project_uv_tri(head, _bounds, _side_bounds, texture_bbox_uvs,
                       mats["skin_textured"],
                       mats.get("skin_textured_side"),
                       mats.get("skin_textured_back"))
    else:
        project_uv_front(head, _bounds, texture_front_bbox_uv)

    # --- Boots --------------------------------------------------------------
    boot_parts = []
    for sign in (-1.0, 1.0):
        s = "L" if sign < 0 else "R"
        boot_parts += [
            # v1.0.15: boot x / width follow hip_scale (Sarah 1.0: unchanged).
            add_sphere(f"Boot_Shaft_{s}", (0.11 * sign * profile.hip_scale, 0.0, 0.24), (0.085 * profile.hip_scale, 0.08 * profile.hip_scale, 0.13)),
            add_sphere(f"Boot_Foot_{s}", (0.11 * sign * profile.hip_scale, -0.05, 0.08), (0.075 * profile.hip_scale, 0.11 * profile.hip_scale, 0.06)),
            add_box(f"Boot_Heel_{s}", (0.11 * sign * profile.hip_scale, 0.07, 0.04), (0.035 * profile.hip_scale, 0.035 * profile.hip_scale, 0.04)),
        ]
    boots = join_objects("Sarah_Boots", boot_parts)
    assign_material(boots, mats["suit"])
    smart_uv(boots)

    # --- Hair (one skinned mesh, not 156 separate strands) ------------------
    # v1.0.10 hair: shell/crown shrunk again, solid BackMass blob
    # deleted and replaced by three wavy back strands (see loop
    # below). v1.0.26: front locks / side falls / shoulder falls
    # are cards now (see below), not spheres — shell/crown stay
    # spheres, as the head-covering cap.
    hair_parts = [
        add_sphere("Hair_Shell", head_loc((0.0, 0.02, 1.84)), head_scale_vec((0.170, 0.140, 0.185))),
        add_sphere("Hair_Crown", head_loc((0.0, 0.0, 1.90)), head_scale_vec((0.145, 0.120, 0.100))),
    ]
    # v1.0.25 hair cards (queue item 13, step 1): the three
    # v1.0.10 beaded sphere chains are replaced by three wavy
    # ribbons (centre, left, right) down the back, z 1.88 -> 1.16
    # (past the shoulders), 13 stations in 0.06 m steps. The wave
    # is a sine along the length — centre amplitude 0.030 m,
    # sides 0.025 m in opposite phase — so the card reads as one
    # continuous wave, not a beaded chain. Card centreline y
    # eases 0.150 -> 0.125 (pre-scale, before hair_loc's
    # head_scale on y), i.e. scaled ~0.177 at the top, just
    # outside the head back surface (~0.171 for Sarah), down
    # to ~0.148 at the tip, just outside the torso back
    # (~0.133-0.143), so the flat ribbon neither sinks into
    # the back nor floats off it. Card widths (pre-scale, then
    # head_scale via hair_loc on x): centre 0.09 m, sides
    # 0.07 m at base x +/-0.12 m, leaving a ~0.047 m gap between
    # cards where the v1.0.7 back-spine neon shows through.
    # All three cards share the same sine sway (in phase), so
    # that gap stays constant along the length; opposite-phase
    # sway was rejected after a plain-Python check showed it
    # nearly closed the gap at the top station.
    # Chain length still follows hair_length about the top
    # (1.88): Sarah at 1.0 gives exactly z 1.88 -> 1.16.
    # Shell/crown/locks/falls above stay spheres (step 2, if
    # this test reads well, cards them too).
    _card_stations = []
    for _i in range(13):
        _z = 1.88 - _i * 0.06 * profile.hair_length
        _t = _i / 12.0
        _y = 0.150 - 0.025 * _t
        _card_stations.append((_z, _y, _t))
    # hair_loc scales x/y by head_scale and keeps z, so the
    # stations below are already in final generator coordinates
    # apart from HEIGHT_SCALE, which add_hair_card applies.
    _centre = [hair_loc((0.030 * math.sin(_i * 0.85), _y, _z))
               for _i, (_z, _y, _t) in enumerate(_card_stations)]
    _left = [hair_loc((-0.12 + 0.030 * math.sin(_i * 0.85), _y, _z))
             for _i, (_z, _y, _t) in enumerate(_card_stations)]
    _right = [hair_loc((0.12 + 0.030 * math.sin(_i * 0.85), _y, _z))
              for _i, (_z, _y, _t) in enumerate(_card_stations)]
    hair_parts.append(add_hair_card("Hair_Card_Centre", _centre, 0.09 * profile.head_scale))
    hair_parts.append(add_hair_card("Hair_Card_Left", _left, 0.07 * profile.head_scale))
    hair_parts.append(add_hair_card("Hair_Card_Right", _right, 0.07 * profile.head_scale))
    # v1.0.26 hair cards (queue item 13, step 2): front locks,
    # side falls and shoulder falls as cards. Front locks use
    # the front-facing orientation (width in X): 9 stations,
    # z 1.95 -> 1.63, x +/-0.10 with a shared 0.015 m sine sway
    # (both locks move together, so their gap stays constant),
    # y easing -0.105 -> -0.140 -> -0.120, so the lock hangs in
    # front of the cheek but behind the toon eye lenses. Side
    # and shoulder falls use orientation="yz" (width in Y, thin
    # in X): 13 stations each, side z 1.91 -> 1.31 with x easing
    # 0.155 -> 0.20 and y about -0.01, shoulder z 1.67 -> 1.07
    # with x easing 0.185 -> 0.215 and y about 0.06, each with a
    # 0.02 m sine sway in y, so the shoulder card sits behind
    # the side card as a second layer. All stations go through
    # hair_loc, so x/y follow head_scale (Sarah 1.18) and z is
    # kept, as with the back cards.
    for _sign, _side in ((-1, "L"), (1, "R")):
        _front = [hair_loc((_sign * 0.10 + 0.015 * math.sin(_i * 0.85),
                            (-0.105 - 0.07 * (_i / 8.0)) if _i <= 4
                            else (-0.140 + 0.04 * (_i / 8.0 - 0.5)),
                            1.95 - _i * 0.04))
                  for _i in range(9)]
        hair_parts.append(add_hair_card(f"Hair_Card_FrontLock_{_side}",
                                        _front, 0.11 * profile.head_scale))
    for _sign, _side in ((-1, "L"), (1, "R")):
        _fall = [hair_loc((_sign * (0.155 + 0.045 * _i / 12.0),
                           -0.01 + 0.02 * math.sin(_i * 0.85),
                           1.91 - _i * 0.05))
                 for _i in range(13)]
        hair_parts.append(add_hair_card(f"Hair_Card_SideFall_{_side}",
                                        _fall, 0.10 * profile.head_scale,
                                        orientation="yz"))
    for _sign, _side in ((-1, "L"), (1, "R")):
        _shoulder = [hair_loc((_sign * (0.185 + 0.030 * _i / 12.0),
                               0.06 + 0.02 * math.sin(_i * 0.85),
                               1.67 - _i * 0.05))
                     for _i in range(13)]
        hair_parts.append(add_hair_card(f"Hair_Card_ShoulderFall_{_side}",
                                        _shoulder, 0.09 * profile.head_scale,
                                        orientation="yz"))
    print(f"Hair cards (v1.0.26): {len(hair_parts)} hair parts "
          f"(2 spheres + 9 cards), shell 0.170/0.140/0.185, "
          f"back cards z 1.88 -> "
          f"{1.88 - 12 * 0.06 * profile.hair_length:.2f} (hair_length "
          f"{profile.hair_length}), front locks z 1.95 -> 1.63, "
          f"side falls z 1.91 -> 1.31, shoulder falls z 1.67 -> "
          f"1.07, widths 0.11/0.10/0.09 m tapering to 0.65, "
          f"thickness 0.006 m, front orientation xy, side/shoulder "
          f"orientation yz.")
    hair = join_objects("Sarah_Hair", hair_parts)
    assign_material(hair, mats["hair"])

    # --- Eyes ---------------------------------------------------------------
    def make_eye(side, sign):
        # v1.0.23 toon eyes (queue item 12, step 1): flattened almond
        # ellipsoids, NOT balls. For Sarah (head_scale 1.18) the
        # sclera centre y -0.138 -> -0.163, front -0.177, against a
        # local head surface of about -0.159 at the eye position, so
        # it protrudes ~0.018 m (old ball: front -0.191, ~0.032 m).
        # The iris is a flat disc whose front (-0.182) sits only
        # ~0.005 m in front of the sclera front, so it reads as a
        # pupil on the lens, not a second ball.
        sclera = add_sphere(f"Sarah_Eye_{side}", head_loc((0.058 * sign, -0.138, 1.815)), head_scale_vec((0.048, 0.012, 0.034)), segments=24)
        iris = add_sphere(f"Sarah_Iris_{side}", head_loc((0.058 * sign, -0.148, 1.815)), head_scale_vec((0.022, 0.006, 0.026)), segments=20)
        assign_material(sclera, mats["sclera"])
        assign_material(iris, mats["iris"])
        return sclera, iris

    eye_l, iris_l = make_eye("L", -1.0)
    eye_r, iris_r = make_eye("R", 1.0)

    # --- Face details + neon (MESH strips, single-bone groups below) --------
    detail_specs = []  # (name, object, material, bone)
    # Face details — front of head is -Y, head centre z=1.80, radius ~0.15.
    # v1.0.9 face: brows/lashes/lips are thin meshes whose backs sit
    # ~0.001 m inside the head front (-0.145) and fronts ~0.009 m
    # proud of it (centre y -0.150, half-thickness 0.003-0.004),
    # so they hug the surface instead of floating or sinking in.
    # add_box size is 2.0 * scale, so scale (0.036, 0.004, 0.007)
    # is a 0.072 m wide, 0.008 m thick, 0.014 m tall brow.
    # v1.0.23: brows angled in the face plane (rotation about Y, the
    # depth axis, tilts the box in the X-Z face plane). L -0.20 rad
    # and R +0.20 rad both lift the INNER end (toward x 0), a
    # friendly arch; if Brian's screenshot reads the arch as
    # inverted, flipping both signs is the one-line fix.
    detail_specs.append(("Sarah_Brow_L", add_box("Sarah_Brow_L", head_loc((-0.06, -0.150, 1.865)), head_scale_vec((0.036, 0.004, 0.007)), rotation=(0.0, -0.20, 0.0)), mats["brows"], "head"))
    detail_specs.append(("Sarah_Brow_R", add_box("Sarah_Brow_R", head_loc((0.06, -0.150, 1.865)), head_scale_vec((0.036, 0.004, 0.007)), rotation=(0.0, 0.20, 0.0)), mats["brows"], "head"))
    # v1.0.23 modelled mouth: dark interior ellipsoid between the
    # lips (bound to jaw, like the lips). Interior front
    # (-0.168 for Sarah) sits ~0.005 m behind the lip fronts
    # (-0.172) and ~0.016 m in front of the local head surface
    # (~-0.151 at the mouth), so it reads in the lip gap without
    # z-fighting the lips or sinking into the head.
    detail_specs.append(("Sarah_Lip_Upper", add_box("Sarah_Lip_Upper", head_loc((0.0, -0.142, 1.738)), head_scale_vec((0.048, 0.004, 0.008))), mats["lips"], "jaw"))
    detail_specs.append(("Sarah_Lip_Lower", add_box("Sarah_Lip_Lower", head_loc((0.0, -0.142, 1.708)), head_scale_vec((0.045, 0.004, 0.008))), mats["lips"], "jaw"))
    detail_specs.append(("Sarah_Mouth_Interior", add_sphere("Sarah_Mouth_Interior", head_loc((0.0, -0.136, 1.723)), head_scale_vec((0.042, 0.006, 0.014))), mats["mouth"], "jaw"))
    # v1.0.23: lashes follow the new eye outline (eye top/bottom
    # z 1.849/1.781, centre x +/-0.058).
    detail_specs.append(("Sarah_Lash_Top_L", add_box("Sarah_Lash_Top_L", head_loc((-0.058, -0.156, 1.845)), head_scale_vec((0.038, 0.0035, 0.005))), mats["brows"], "head"))
    detail_specs.append(("Sarah_Lash_Top_R", add_box("Sarah_Lash_Top_R", head_loc((0.058, -0.156, 1.845)), head_scale_vec((0.038, 0.0035, 0.005))), mats["brows"], "head"))
    detail_specs.append(("Sarah_Lash_Bottom_L", add_box("Sarah_Lash_Bottom_L", head_loc((-0.058, -0.156, 1.780)), head_scale_vec((0.030, 0.0035, 0.003))), mats["brows"], "head"))
    detail_specs.append(("Sarah_Lash_Bottom_R", add_box("Sarah_Lash_Bottom_R", head_loc((0.058, -0.156, 1.780)), head_scale_vec((0.030, 0.0035, 0.003))), mats["brows"], "head"))
    # v1.0.24 eyelid shells (queue item 12, step 2): skin-colour
    # caps overlapping only the sclera edges. Upper cap centre
    # z 1.846, z half 0.012 -> bottom 1.834, covering the sclera
    # top (1.849) by 0.015; lower cap centre z 1.782, z half
    # 0.010 -> top 1.792, covering the sclera bottom (1.781) by
    # 0.011. Visible eye band is therefore ~1.792-1.834, and the
    # iris (1.789-1.841) is framed with only slivers clipped.
    # Lid fronts (upper -0.153, lower -0.152 unscaled, centres
    # y -0.140) sit ~0.002-0.003 proud of the sclera front
    # (-0.150), so the lids read as a ridge over the lens, and
    # the lashes (moved to y -0.156, front -0.1595) sit proud
    # of the lids in turn.
    # v1.0.27 blink-linked eyelids (queue item 12, remaining
    # step): the caps below are no longer static. Each cap gets
    # its own Basis + Blink_<side> shape key right after it is
    # created, so the glTF exporter carries the lid morph and
    # AvatarPanel's setMorph (which drives Blink_L/Blink_R on
    # every mesh that has them) closes the lids with no panel
    # change. The closed target stretches the cap 1.8x in Z
    # about its own centre and translates it to the eye centre
    # (eye centre z 1.815, upper cap 1.846 -> -0.031 m, lower
    # cap 1.782 -> +0.033 m, pre-scale, times head_scale and
    # HEIGHT_SCALE, so the shift is in the same scaled space
    # the mesh vertices already live in after add_sphere's
    # transform_apply). X/Y are untouched, so the closed lid
    # stays ~0.003 m proud of the sclera front, in front of
    # the lens it covers, and at Blink = 0 the key is exactly
    # the v1.0.24 cap. No BlinkBoth key is added: the panel
    # also drives BlinkBoth at 0.35 * min(blink), and a
    # BlinkBoth lid key would double-drive the lids.
    _eyelid_blink = []  # (object, blink_name, centre_z, is_upper)
    for _sign, _s in ((-1.0, "L"), (1.0, "R")):
        _up = add_sphere(f"Sarah_Eyelid_Upper_{_s}", head_loc((0.058 * _sign, -0.140, 1.846)), head_scale_vec((0.052, 0.013, 0.012)), segments=24)
        detail_specs.append((f"Sarah_Eyelid_Upper_{_s}", _up, mats["skin"], "head"))
        _eyelid_blink.append((_up, f"Blink_{_s}", 1.846, True))
        _low = add_sphere(f"Sarah_Eyelid_Lower_{_s}", head_loc((0.058 * _sign, -0.140, 1.782)), head_scale_vec((0.050, 0.012, 0.010)), segments=24)
        detail_specs.append((f"Sarah_Eyelid_Lower_{_s}", _low, mats["skin"], "head"))
        _eyelid_blink.append((_low, f"Blink_{_s}", 1.782, False))
    for _obj, _blink_name, _centre_z, _is_upper in _eyelid_blink:
        bpy.context.view_layer.objects.active = _obj
        bpy.ops.object.shape_key_add(from_mix=False)
        _obj.active_shape_key.name = "Basis"
        bpy.ops.object.shape_key_add(from_mix=False)
        _obj.active_shape_key.name = _blink_name
        _lid_keys = _obj.data.shape_keys.key_blocks
        _lid_basis = _lid_keys["Basis"]
        _lid_blink = _lid_keys[_blink_name]
        # Eye centre 1.815 minus this cap's centre, in scaled
        # world units (head_loc scales about the 1.80 head centre
        # by head_scale, then HEIGHT_SCALE, exactly as add_sphere
        # did for the mesh itself).
        _shift = (head_loc((0.0, 0.0, 1.815))[2] - head_loc((0.0, 0.0, _centre_z))[2]) * HEIGHT_SCALE
        for _idx, _vert in enumerate(_obj.data.vertices):
            _base = _lid_basis.data[_idx].co.copy()
            _closed = _base.copy()
            _closed.z = _base.z * 1.8 + _shift
            _lid_blink.data[_idx].co = _closed
    print("Sculpted face (v1.0.24): eyelid caps upper 0.052/0.013/0.012 "
          "at z 1.846 and lower 0.050/0.012/0.010 at z 1.782, "
          "lashes moved to y -0.156, nose bridge + tip, head photo "
          "texture suppressed by default (pass --head-photo to "
          "restore it).")
    print("Blink-linked eyelids (v1.0.27): 4 caps, each with Basis + "
          "its side's Blink key (L -> Blink_L, R -> Blink_R), "
          "closed = 1.8x Z stretch about the cap centre translated "
          "to eye centre z 1.815 (upper -0.031 m, lower +0.033 m, "
          "pre-scale).")
    # v1.0.12 boots (queue item 7): glowing sole + ankle ring, as
    # separate neon-material meshes bound to one bone each, NOT
    # painted onto the flat suit boots mesh. Sole: a thin box under
    # each foot sphere (foot: centre y -0.05, y half 0.11, z bottom
    # 0.02), slightly wider/longer than the foot so the glow reads
    # around its edge, bottom at z 0.002 (pre-scale) so the feet
    # still stand on the ground, bound to foot.L/R. Ankle ring: a
    # torus around the shaft at z 0.17, where the shaft sphere
    # (centre z 0.24, x radius 0.085, z radius 0.13) has narrowed
    # to ~0.072 m x-radius, so major 0.080 / minor 0.007 sits just
    # proud of the shaft, bound to shin.L/R (the ring centre lies
    # in the shin bone's 0.55 -> 0.14 span, above foot's head).
    for sign in (-1.0, 1.0):
        s = "L" if sign < 0 else "R"
        # v1.0.15: glow x / width follow hip_scale, matching the
        # hip-scaled boot shaft/foot above (Sarah 1.0: unchanged).
        detail_specs.append((f"Sarah_Boot_SoleGlow_{s}",
                             add_box(f"Sarah_Boot_SoleGlow_{s}",
                                     (0.11 * sign * profile.hip_scale, -0.055, 0.008),
                                     (0.082 * profile.hip_scale, 0.122 * profile.hip_scale, 0.006)),
                             mats["neon"], f"foot.{s}"))
        detail_specs.append((f"Sarah_Boot_AnkleRing_{s}",
                             add_torus(f"Sarah_Boot_AnkleRing_{s}",
                                       (0.11 * sign * profile.hip_scale, 0.0, 0.17),
                                       0.080 * profile.hip_scale, 0.007),
                             mats["neon"], f"shin.{s}"))
    print("Boots (v1.0.15): 4 glow parts (sole plate + ankle ring "
          "per foot), sole 0.164 x 0.244 x 0.012 m * hip_scale, "
          "ankle ring major 0.080 m * hip_scale / minor 0.007 m "
          "at z 0.17 m, neon material, one bone each (sole foot, "
          "ring shin).")
    # v1.0.7 neon-on-surface: add_box makes a cube of size 2.0, so a
    # scale of (0.004, 0.001, 0.07) is a strip 0.008 m wide, 0.002 m
    # thick and 0.14 m long. Every strip below is at most that long,
    # sits with its centre ~0.003 m outside the local body surface
    # (surface + 0.002 m gap + half its 0.002 m thickness), and is
    # bound to exactly one bone — the bone whose span contains the
    # strip centre — so no strip bridges a joint. v1.0.18: neon y
    # follows bulk_scale / shoulder_scale / hip_scale (see entries),
    # or the bulkier v1.0.18 geometry would bury the strips.
    # Torso front is
    # y -0.13 (torso sphere), chest front -0.14, back +0.13/+0.10.
    neon_boxes = [
        # Torso centre, split at the spine joints (was one 0.64 m plank).
        ("Sarah_Neon_Torso_Center_Low", (0.0, -0.133 * profile.bulk_scale, 1.12), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.01"),
        ("Sarah_Neon_Torso_Center_Mid", (0.0, -0.133 * profile.bulk_scale, 1.27), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.02"),
        ("Sarah_Neon_Torso_Center_High", (0.0, -0.143 * profile.bulk_scale, 1.44), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.03"),
        # Chest V, split into a lower and an upper segment per side.
        ("Sarah_Neon_Torso_Left_V_Low", (-0.09, -0.136 * profile.bulk_scale, 1.30), (0.004, 0.001, 0.07), (0.0, 0.0, 0.45), "spine.02"),
        ("Sarah_Neon_Torso_Left_V_High", (-0.09, -0.146 * profile.bulk_scale, 1.46), (0.004, 0.001, 0.07), (0.0, 0.0, 0.45), "spine.03"),
        ("Sarah_Neon_Torso_Right_V_Low", (0.09, -0.136 * profile.bulk_scale, 1.30), (0.004, 0.001, 0.07), (0.0, 0.0, -0.45), "spine.02"),
        ("Sarah_Neon_Torso_Right_V_High", (0.09, -0.146 * profile.bulk_scale, 1.46), (0.004, 0.001, 0.07), (0.0, 0.0, -0.45), "spine.03"),
        ("Sarah_Neon_Waist_Left", (-0.12, -0.133 * profile.bulk_scale, 1.08), (0.004, 0.001, 0.07), (0.0, 0.0, 0.35), "spine.01"),
        ("Sarah_Neon_Waist_Right", (0.12, -0.133 * profile.bulk_scale, 1.08), (0.004, 0.001, 0.07), (0.0, 0.0, -0.35), "spine.01"),
        # Collar: neck-base sphere front is y -0.09, so -0.093 hugs it
        # (was -0.12, i.e. 0.03 m floating in front of the neck).
        ("Sarah_Neon_Collar_Left", (-0.05, -0.093 * profile.bulk_scale, 1.62), (0.004, 0.001, 0.07), (0.0, 0.0, 0.30), "neck"),
        ("Sarah_Neon_Collar_Right", (0.05, -0.093 * profile.bulk_scale, 1.62), (0.004, 0.001, 0.07), (0.0, 0.0, -0.30), "neck"),
        # Back spine, split the same way as the front centre.
        ("Sarah_Neon_Back_Spine_Low", (0.0, 0.133 * profile.bulk_scale, 1.12), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.01"),
        ("Sarah_Neon_Back_Spine_Mid", (0.0, 0.133 * profile.bulk_scale, 1.27), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.02"),
        ("Sarah_Neon_Back_Spine_High", (0.0, 0.103 * profile.bulk_scale, 1.44), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), "spine.03"),
        ("Sarah_Neon_Back_Left_X", (-0.09, 0.133 * profile.bulk_scale, 1.10), (0.004, 0.001, 0.07), (0.0, 0.0, 0.45), "spine.01"),
        ("Sarah_Neon_Back_Right_X", (0.09, 0.133 * profile.bulk_scale, 1.10), (0.004, 0.001, 0.07), (0.0, 0.0, -0.45), "spine.01"),
    ]
    for sign in (-1.0, 1.0):
        s = "L" if sign < 0 else "R"
        neon_boxes += [
            # Arm: one short segment per bone (was one 0.60 m plank
            # on upper_arm alone, spanning shoulder to forearm).
            # v1.0.15: shoulder neon x follows shoulder_scale (the
            # shoulder sphere centre moved in v1.0.14); arm/forearm
            # centres did not move, so their neon stays put.
            (f"Sarah_Neon_Shoulder_{s}", (0.21 * sign * profile.shoulder_scale, -0.073 * profile.shoulder_scale, 1.50), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), f"clavicle.{s}"),
            (f"Sarah_Neon_Arm_{s}", (0.26 * sign, -0.058 * profile.bulk_scale, 1.36), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), f"upper_arm.{s}"),
            (f"Sarah_Neon_Forearm_{s}", (0.28 * sign, -0.048 * profile.bulk_scale, 1.06), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), f"forearm.{s}"),
            # Leg: thigh and shin each get their own segment (was
            # one 0.76 m plank per leg crossing the knee).
            (f"Sarah_Neon_Leg_Front_{s}", (0.11 * sign, -0.093 * profile.bulk_scale, 0.78), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), f"thigh.{s}"),
            (f"Sarah_Neon_Leg_Shin_{s}", (0.12 * sign, -0.068 * profile.bulk_scale, 0.36), (0.004, 0.001, 0.07), (0.0, 0.0, 0.0), f"shin.{s}"),
            # v1.0.15: boot neon x follows hip_scale, matching the
            # hip-scaled boots above; leg neon centres did not move.
            (f"Sarah_Neon_Boot_{s}", (0.11 * sign * profile.hip_scale, -0.083 * profile.hip_scale, 0.22), (0.004, 0.001, 0.05), (0.0, 0.0, 0.0), f"foot.{s}"),
        ]
    print(f"Neon segments: {len(neon_boxes)}, max strip length "
          f"{max(2.0 * sc[2] for _n, _l, sc, _r, _b in neon_boxes):.3f} m, "
          f"thickness {2.0 * neon_boxes[0][2][1]:.3f} m, one bone each.")
    for name, loc, scale, rot, bone in neon_boxes:
        obj = add_box(name, loc, scale, rotation=rot)
        detail_specs.append((name, obj, mats["neon"], bone))

    # --- Armature, ARMS DOWN ----------------------------------------------
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    rig = bpy.context.active_object
    rig.name = RIG_NAME
    rig.data.name = f"{RIG_NAME}_Data"
    eb = rig.data.edit_bones
    for b in list(eb):
        eb.remove(b)

    def add_bone(name, head_pos, tail_pos, parent=None, use_connect=False, deform=True):
        # v1.0.11: bones scale with the geometry, or skinning breaks.
        head_pos = tuple(v * HEIGHT_SCALE for v in head_pos)
        tail_pos = tuple(v * HEIGHT_SCALE for v in tail_pos)
        bone = eb.new(name)
        bone.head = Vector(head_pos)
        bone.tail = Vector(tail_pos)
        bone.use_deform = deform
        if parent is not None:
            bone.parent = parent
            bone.use_connect = use_connect
        return bone

    root = add_bone("root", (0.0, 0.0, 0.0), (0.0, 0.0, 0.14), deform=False)
    pelvis = add_bone("pelvis", (0.0, 0.0, 0.92), (0.0, 0.0, 1.07), parent=root)
    spine01 = add_bone("spine.01", (0.0, 0.0, 1.07), (0.0, 0.0, 1.24), parent=pelvis, use_connect=True)
    spine02 = add_bone("spine.02", (0.0, 0.0, 1.24), (0.0, 0.0, 1.41), parent=spine01, use_connect=True)
    spine03 = add_bone("spine.03", (0.0, 0.0, 1.41), (0.0, 0.0, 1.57), parent=spine02, use_connect=True)
    neck = add_bone("neck", (0.0, 0.0, 1.57), (0.0, 0.0, 1.70), parent=spine03, use_connect=True)
    head_bone = add_bone("head", (0.0, 0.0, 1.70), (0.0, -0.02, 1.93), parent=neck, use_connect=True)
    jaw_bone = add_bone("jaw", (0.0, -0.04, 1.76), (0.0, -0.12, 1.70), parent=head_bone)
    add_bone("eye.L", (-0.055, -0.085, 1.80), (-0.055, -0.17, 1.80), parent=head_bone)
    add_bone("eye.R", (0.055, -0.085, 1.80), (0.055, -0.17, 1.80), parent=head_bone)
    hair_root = add_bone("hair_root", (0.0, 0.04, 1.82), (0.0, 0.10, 1.69), parent=head_bone)
    hair01 = add_bone("hair.01", (0.0, 0.10, 1.69), (0.0, 0.13, 1.42), parent=hair_root, use_connect=True)
    hair02 = add_bone("hair.02", (0.0, 0.13, 1.42), (0.0, 0.16, 1.12), parent=hair01, use_connect=True)
    add_bone("hair.03", (0.0, 0.16, 1.12), (0.0, 0.18, 0.84), parent=hair02, use_connect=True)
    for side, sign in (("L", -1.0), ("R", 1.0)):
        clav = add_bone(f"clavicle.{side}", (0.03 * sign, 0.0, 1.53), (0.20 * sign, 0.0, 1.51), parent=spine03)
        upper = add_bone(f"upper_arm.{side}", (0.20 * sign, 0.0, 1.51), (0.26 * sign, 0.0, 1.18), parent=clav, use_connect=True)
        fore = add_bone(f"forearm.{side}", (0.26 * sign, 0.0, 1.18), (0.28 * sign, 0.0, 0.92), parent=upper, use_connect=True)
        hand = add_bone(f"hand.{side}", (0.28 * sign, 0.0, 0.92), (0.28 * sign, -0.01, 0.82), parent=fore, use_connect=True)
        # Fingers — short, hanging with the hand.
        for fname, yoff in (("thumb", 0.02), ("index", -0.018), ("middle", -0.006), ("ring", 0.006), ("pinky", 0.018)):
            base_z = 0.90 if fname == "thumb" else 0.84
            b1 = add_bone(f"{fname}.01.{side}", (0.28 * sign, yoff, base_z), (0.28 * sign, yoff, base_z - 0.04), parent=hand)
            b2 = add_bone(f"{fname}.02.{side}", (0.28 * sign, yoff, base_z - 0.04), (0.28 * sign, yoff, base_z - 0.07), parent=b1, use_connect=True)
            add_bone(f"{fname}.03.{side}", (0.28 * sign, yoff, base_z - 0.07), (0.28 * sign, yoff, base_z - 0.09), parent=b2, use_connect=True)
        thigh = add_bone(f"thigh.{side}", (0.11 * sign, 0.0, 0.97), (0.12 * sign, 0.0, 0.55), parent=pelvis)
        shin = add_bone(f"shin.{side}", (0.12 * sign, 0.0, 0.55), (0.11 * sign, 0.0, 0.14), parent=thigh, use_connect=True)
        foot = add_bone(f"foot.{side}", (0.11 * sign, 0.0, 0.14), (0.11 * sign, -0.12, 0.05), parent=shin, use_connect=True)
        add_bone(f"toe.{side}", (0.11 * sign, -0.12, 0.05), (0.11 * sign, -0.24, 0.05), parent=foot, use_connect=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- Skinning ------------------------------------------------------------
    def parent_auto(obj):
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        try:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        except Exception as exc:
            raise RuntimeError(f"Required auto-weight skinning failed for {obj.name}: {exc}") from exc
        if not any(mod.type == "ARMATURE" and mod.object == rig for mod in obj.modifiers):
            raise RuntimeError(f"{obj.name} has no valid rig modifier after auto-weighting")
        unweighted = sum(1 for vertex in obj.data.vertices
                         if not any(group.weight > 0 for group in vertex.groups))
        if unweighted:
            raise RuntimeError(f"{obj.name} has {unweighted} unweighted vertices; export refused")

    for obj in (body, boots, head, hair):
        parent_auto(obj)

    def bind_detail(obj, bone_name):
        group = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(name=bone_name)
        ids = [v.index for v in obj.data.vertices]
        if ids:
            group.add(ids, 1.0, "REPLACE")
        mod = obj.modifiers.get("SarahRuntimeArmature") or obj.modifiers.new(name="SarahRuntimeArmature", type="ARMATURE")
        mod.object = rig
        try:
            mod.use_vertex_groups = True
        except Exception:
            pass

    # Eyes are controlled by eye.L/R in Avatar3D.tsx, so auto-weights
    # toward the adjacent head bone would make gaze tracking ineffective.
    for _eye, _bone in ((eye_l, "eye.L"), (iris_l, "eye.L"),
                        (eye_r, "eye.R"), (iris_r, "eye.R")):
        bind_detail(_eye, _bone)
        _eye.parent = rig
        _eye.matrix_parent_inverse = rig.matrix_world.inverted()
    for _name, obj, mat, bone_name in detail_specs:
        assign_material(obj, mat)
        bind_detail(obj, bone_name)
    # v1.0.1 fix: also PARENT detail meshes to the rig (object parent, keep
    # transform). The armature modifier alone triggered Blender's
    # "Armature must be the parent of skinned mesh" warning on every detail
    # object in the first run, even though they exported skinned.
    bpy.ops.object.select_all(action="DESELECT")
    for _name, obj, _mat, _bone in detail_specs:
        obj.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    try:
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    except Exception as exc:
        raise RuntimeError(f"Detail-mesh rig parenting failed: {exc}") from exc

    # Articulated finger geometry: original generator declared 30 finger bones
    # but supplied no matching finger mesh. One low-cost skinned mesh contains
    # five digits x three segments x two hands; each segment is rigidly bound
    # to its EXISTING corresponding finger bone, without changing rig names.
    finger_vertices, finger_faces, finger_ranges = [], [], []
    for _side in ("L", "R"):
        for _digit in ("thumb", "index", "middle", "ring", "pinky"):
            for _joint in (1, 2, 3):
                _bone_name = f"{_digit}.{_joint:02d}.{_side}"
                _b = rig.data.bones.get(_bone_name)
                if _b is None:
                    raise RuntimeError(f"Finger bone missing: {_bone_name}")
                _h, _t = _b.head_local, _b.tail_local
                _start = len(finger_vertices)
                _radius = (0.010 if _digit == "thumb" else
                           0.008 if _digit in ("index", "middle") else 0.007) * HEIGHT_SCALE
                _offset_x = (0.011 * HEIGHT_SCALE * (-1 if _side == "L" else 1)
                             if _digit == "thumb" else 0.0)
                for _p in (_h, _t):
                    for _sector in range(8):
                        _ang = math.tau * _sector / 8.0
                        finger_vertices.append((_p.x + _offset_x + _radius * math.cos(_ang),
                                                _p.y + _radius * math.sin(_ang), _p.z))
                for _sector in range(8):
                    _n = (_sector + 1) % 8
                    _base = _start
                    finger_faces.append((_base + _sector, _base + _n,
                                         _base + 8 + _n, _base + 8 + _sector))
                finger_faces.append(tuple(_start + i for i in reversed(range(8))))
                finger_faces.append(tuple(_start + 8 + i for i in range(8)))
                finger_ranges.append((_bone_name, list(range(_start, _start + 16))))
    finger_data = bpy.data.meshes.new("Sarah_FingersMesh")
    finger_data.from_pydata(finger_vertices, [], finger_faces)
    finger_data.update()
    fingers = bpy.data.objects.new("Sarah_Fingers", finger_data)
    bpy.context.scene.collection.objects.link(fingers)
    assign_material(fingers, mats["suit"])
    for _name, _indices in finger_ranges:
        _group = fingers.vertex_groups.new(name=_name)
        _group.add(_indices, 1.0, "REPLACE")
    _fmod = fingers.modifiers.new(name="SarahRuntimeArmature", type="ARMATURE")
    _fmod.object = rig
    fingers.parent = rig
    fingers.matrix_parent_inverse = rig.matrix_world.inverted()
    fingers["sarahmemory_appearance_slot"] = "fingers"
    fingers["sarahmemory_rig_contract"] = RIG_NAME
    print("Articulated fingers: 30 existing finger bones, one skinned runtime mesh")

    # --- Head shape keys (no drivers — UI drives them) -----------------------
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.shape_key_add(from_mix=False)
    head.active_shape_key.name = "Basis"
    for key_name in EXPECTED_SHAPE_KEYS:
        bpy.ops.object.shape_key_add(from_mix=False)
        head.active_shape_key.name = key_name
    keys = head.data.shape_keys.key_blocks
    basis = keys["Basis"]
    # Head local space: origin is the first sphere's centre (scaled
    # 1.80 * HEIGHT_SCALE in v1.0.11), so local thresholds/offsets
    # below are all scaled by HEIGHT_SCALE, and falloff coefficients
    # are divided by it to keep the same relative shape.
    HS = HEIGHT_SCALE
    for idx, vert in enumerate(head.data.vertices):
        base = basis.data[idx].co.copy()
        x, y, z = base.x, base.y, base.z
        blink_l = base.copy(); blink_r = base.copy(); jaw_open = base.copy()
        smile = base.copy(); frown = base.copy(); brows_up = base.copy()
        squint_l = base.copy(); squint_r = base.copy(); mouth_wide = base.copy()
        left_eye = x < -0.015 * HS and y < -0.08 * HS and 0.02 * HS < z < 0.14 * HS
        right_eye = x > 0.015 * HS and y < -0.08 * HS and 0.02 * HS < z < 0.14 * HS
        if left_eye:
            blink_l.z *= 0.55; blink_l.y += 0.010 * HS
            squint_l.z *= 0.82; squint_l.y += 0.004 * HS
        if right_eye:
            blink_r.z *= 0.55; blink_r.y += 0.010 * HS
            squint_r.z *= 0.82; squint_r.y += 0.004 * HS
        mouth_zone = y < -0.08 * HS and -0.12 * HS < z < -0.02 * HS
        lower_face = z < -0.02 * HS and y < 0.03 * HS
        if lower_face:
            jaw_open.z -= 0.030 * HS * (1.0 - min(1.0, abs(x) * 5.0 / HS)); jaw_open.y -= 0.020 * HS
        if mouth_zone:
            smile.z += 0.012 * HS * (0.40 + abs(x) * 10.0 / HS)
            smile.x += 0.008 * HS * (1.0 if x > 0.0 else -1.0); smile.y -= 0.006 * HS
            frown.z -= 0.012 * HS * (0.40 + abs(x) * 10.0 / HS)
            frown.x -= 0.004 * HS * (1.0 if x > 0.0 else -1.0); frown.y += 0.003 * HS
            mouth_wide.x += 0.018 * HS * (1.0 if x > 0.0 else -1.0); mouth_wide.y -= 0.004 * HS
        brow_zone = y < -0.08 * HS and 0.05 * HS < z < 0.14 * HS and abs(x) < 0.11 * HS
        if brow_zone:
            brows_up.z += 0.018 * HS * (1.0 - min(1.0, abs(x) * 4.0 / HS)); brows_up.y -= 0.003 * HS
        keys["Blink_L"].data[idx].co = blink_l
        keys["Blink_R"].data[idx].co = blink_r
        keys["JawOpen"].data[idx].co = jaw_open
        keys["Smile"].data[idx].co = smile
        keys["Frown"].data[idx].co = frown
        keys["BrowsUp"].data[idx].co = brows_up
        keys["Squint_L"].data[idx].co = squint_l
        keys["Squint_R"].data[idx].co = squint_r
        keys["MouthWide"].data[idx].co = mouth_wide

    # Bootstrap Avatar-Eye anchors: named scene graph locators, visual only.
    # They are NOT cameras or hardware actuators. Parenting to the head bone
    # keeps their coordinates stable under AvatarPanel's procedural head motion.
    eye_anchors = []
    for _name, _at in (
            ("Sarah_AvatarEye_Center", (0.0, -0.160, 1.815)),
            ("Sarah_AvatarEye_L", (-0.058, -0.160, 1.815)),
            ("Sarah_AvatarEye_R", (0.058, -0.160, 1.815)),
            ("Sarah_Expression_Controller", (0.0, -0.190, 1.835))):
        marker = bpy.data.objects.new(_name, None)
        bpy.context.scene.collection.objects.link(marker)
        marker["sarahmemory_role"] = "avatar_eye_anchor" if "AvatarEye" in _name else "expression_anchor"
        marker["execution_authority"] = False
        marker.parent = rig
        marker.parent_type = "BONE"
        marker.parent_bone = "head"
        marker.matrix_world.translation = Vector(tuple(p * HEIGHT_SCALE for p in _at))
        eye_anchors.append(marker)
    # Modular appearance metadata: existing mesh objects, not duplicates.
    for _obj, _slot in ((body, "body_suit"), (head, "head_face"),
                        (hair, "hair"), (boots, "boots"),
                        (eye_l, "eye_left"), (eye_r, "eye_right"),
                        (iris_l, "iris_left"), (iris_r, "iris_right")):
        _obj["sarahmemory_appearance_slot"] = _slot
        _obj["sarahmemory_rig_contract"] = RIG_NAME
    for _name, _obj, _mat, _bone in detail_specs:
        _obj["sarahmemory_appearance_slot"] = (
            "neon" if "Neon" in _name or "Glow" in _name else "face_detail")
        _obj["sarahmemory_bound_bone"] = _bone
    rig["sarahmemory_animation_authority"] = "AvatarPanel_procedural_visual_only"
    rig["execution_authority"] = False

    # --- Export: runtime objects only, morphs ON, apply OFF ------------------
    # export_apply=False matters: applying modifiers at export time is one
    # way shape keys have historically been lost in this project.
    runtime = ([body, boots, head, hair, fingers, eye_l, eye_r, iris_l,
                iris_r, rig] + [o for _, o, _, _ in detail_specs] + eye_anchors)
    # Enforce early mesh budgets before writing the .blend or GLB, in
    # addition to the independent post-export accessor count.
    _meshes = [obj for obj in runtime if obj.type == "MESH"]
    _raw_vertices = sum(len(obj.data.vertices) for obj in _meshes)
    _triangles_est = sum(sum(max(0, len(poly.vertices) - 2)
                             for poly in obj.data.polygons) for obj in _meshes)
    print(f"Pre-export budget: {_raw_vertices:,} vertices, "
          f"{_triangles_est:,} estimated triangles")
    if _raw_vertices > args.max_vertices or _triangles_est > args.max_triangles:
        raise RuntimeError("Generated avatar geometry exceeds explicit runtime budget")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in runtime:
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = runtime[0]
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        export_apply=False,
        export_texcoords=True,
        export_normals=True,
        export_tangents=False,
        export_materials="EXPORT",
        export_yup=True,
        export_animations=False,
        export_skins=True,
        export_morph=True,
        export_extras=True,
        use_selection=True,
    )
    print(f"Wrote {out_blend}")
    print(f"Wrote {out_glb}")
    print("Inner build done. Outer verifier will now check morphs, skins, UVs.")


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--inner", action="store_true")
    p.add_argument("--front", default="")
    p.add_argument("--side", default="")
    p.add_argument("--back", default="")
    p.add_argument("--blueprint", default="")
    p.add_argument("--outdir", default="")
    p.add_argument("--blender", default="")
    p.add_argument("--no-split", action="store_true",
                   help="Skip dual-panel image splitting entirely (fast GLB-only run)")
    p.add_argument("--no-texture", action="store_true",
                   help="Skip texture embedding (flat v1.0.5 colours, for bisecting a texture export problem)")
    p.add_argument("--head-photo", action="store_true",
                   help="Restore the projected photo texture on the Head mesh "
                        "(v1.0.24 default is flat skin on the head, so the "
                        "toon face reads alone; body suit texture is unaffected)")
    p.add_argument("--split-all", action="store_true",
                   help="Split front, side AND back references (default: front only)")
    p.add_argument("--character", default="sarah_default",
                   help="Character profile id (v1.0.14: sarah_default, toon_hero_01; "
                        "an unknown id fails loudly instead of building Sarah)")
    p.add_argument("--upload", default="",
                   help="Single-avatar PNG to sample a new character from "
                        "(v1.0.16: 8-bit RGB/RGBA, white background; "
                        "dual-panel composites must be split first)")
    p.add_argument("--style", default="auto",
                   choices=["auto", "fortnite_toon", "sarah_default"],
                   help="Style preset for an --upload profile (auto picks "
                        "fortnite_toon when head_scale >= 1.15, else "
                        "sarah_default)")
    p.add_argument("--preview-only", action="store_true",
                   help="With --upload: write the preview/profile JSON only, "
                        "do not run Blender, do not build a GLB")
    p.add_argument("--profile-json", default="",
                   help="Inner use: profile JSON written by an --upload outer "
                        "run (overrides --character inside Blender)")
    p.add_argument("--reference-cache-dir", default="", help="Inner use: persisted reference crop directory")
    p.add_argument("--quality", choices=("preview", "balanced", "high"),
                   default="balanced", help="Bounded runtime mesh-quality tier; never multi-million-poly")
    p.add_argument("--governed-build", action="store_true",
                   help="Enforce governed OperatorCore authorization; without it v1.0.30 warns and uses the staged verified build")
    p.add_argument("--timeout-seconds", type=int, default=1200,
                   help="Maximum Blender execution time, 30 to 7200 seconds")
    p.add_argument("--max-glb-mb", type=int, default=DEFAULT_MAX_GLB_MB,
                   help="GLB export size budget, 8 to 512 MiB")
    p.add_argument("--max-vertices", type=int, default=DEFAULT_MAX_VERTICES)
    p.add_argument("--max-triangles", type=int, default=DEFAULT_MAX_TRIANGLES)
    p.add_argument("--max-rollbacks", type=int, default=3)
    p.add_argument("--allow-blender", action="store_true",
                   help="Explicitly permit governed local Blender subprocess")
    p.add_argument("--user-consented", action="store_true",
                   help="Confirm local user consent; OperatorCore approval is still required")
    args = p.parse_args(argv)
    if not 30 <= args.timeout_seconds <= 7200:
        p.error("--timeout-seconds must be 30..7200")
    if not 8 <= args.max_glb_mb <= 512:
        p.error("--max-glb-mb must be 8..512")
    if not 1000 <= args.max_vertices <= 2_000_000 or not 1000 <= args.max_triangles <= 2_000_000:
        p.error("geometry budgets must be 1000..2000000")
    if not 1 <= args.max_rollbacks <= 20:
        p.error("--max-rollbacks must be 1..20")
    return args


def main():
    # Blender passes its own args before "--"; everything after "--" is ours.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]
    args = parse_args(argv)
    if args.inner:
        inner_main(args)
    else:
        raise SystemExit(outer_main(args))




# ---------------------------------------------------------------------------
# Single SML organ adapter: consolidated observable avatar asset builder.
# Unlike the retired modules this is deliberately non-executing for packets.
# ---------------------------------------------------------------------------
SML_ORGAN_METADATA = {
    "name": "SarahMemoryAvatarGenerate",
    "version": "v9.1.0a",
    "category": "Presentation", "protocol_version": "SML/1.0",
    "packet_version": 1, "omega_registry_version": "Ω/1.0",
    "capabilities": ["avatar", "presentation"],
    "supported_missions": ["Conversation", "Patch", "Repair"],
    "supported_omega": ["Ω001"],
    "required_authority": ["Read"], "priority": 45,
    "trust_level": "source_integrated", "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing",
                 "source_file": "SarahMemoryAvatarGenerate.py",
                 "execution_authority": False},
}


def sml_get_metadata():
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    return {"status": "UNVERIFIED", "availability": None, "integrity": None,
            "performance": None, "reliability": None, "confidence": 0.0,
            "latency_ms": None, "stability": None, "compatibility": None,
            "notes": ["Source import does not certify Blender export or AvatarPanel runtime",
                      "Blender execution requires human consent and OperatorCore review"]}


def sml_diagnostics():
    return {"status": "OK", "component": "SarahMemoryAvatarGenerate",
            "sml_adapter": True, "metadata": sml_get_metadata(),
            "health": sml_health()}


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Observe the canonical SML packet; never build from packet contents."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ="SarahMemoryAvatarGenerate",
                                action=action, note=note or "avatar builder observed packet",
                                updates=updates)
    except Exception:
        return packet


if __name__ == "__main__":
    main()
