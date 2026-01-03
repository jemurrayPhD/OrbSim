# Packaging OrbSim with pyside6-deploy (Windows stub)

This project is ready to be bundled with `pyside6-deploy`. The deployment flow below is a stub for Windows packaging and can be adapted for macOS/Linux.

## 1) Generate resources
Compile the Qt resources so `:/icons/*` assets are embedded:

```bash
pyside6-rcc src/orbsim/ui/resources.qrc -o src/orbsim/resources_rc.py
```

## 2) Create the deployment spec
```bash
pyside6-deploy --init
```

## 3) Update the spec
Ensure the generated spec includes:
- Entry point: `orbsim.app:main`
- Data files: `src/orbsim/ui/resources.qrc`
- Resource Python module: `src/orbsim/resources_rc.py`

## 4) Build the executable
```bash
pyside6-deploy
```

## Notes
- Runtime data (databases/images) are stored in `QStandardPaths.AppDataLocation`.
- All new icons are routed through Qt resources (see `resources.qrc`).
