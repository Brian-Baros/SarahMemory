# SarahMemory AiOS Default SoftPack (sm_ai_os_default_softpack)

This is a default distribution bundle of baseline software integrations for SarahMemory AiOS.

## Install
Copy this folder to:

../data/addons/sm_ai_os_default_softpack/

## Bundle contents
- providers/os_default        (open URL/file/folder using OS defaults)
- scanners/software_scanner_light (detect common browsers/media/office; writes data/registry/software_index.json)
- router/capability_router    (DB-first then cache then OS-default routing decisions)

## Expected loader behavior
Your AddonLauncher should:
1) Detect addon_bundle manifest at this folder
2) Enumerate subfolders listed in includes
3) For each included subfolder, load its addon.py and register its actions

If the loader does not yet support bundles, you can still load these sub-addons directly by pointing the loader at the subfolder paths.
