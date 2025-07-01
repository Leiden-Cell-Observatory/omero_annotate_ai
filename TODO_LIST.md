# Project TODO List - Codebase Improvement Plan

## Phase 1: Essential Code Organization (1-2 days)

### ✅ Completed
- [x] **Phase 1: Fix pyproject.toml dependencies** - Fixed ezomero version conflict to 3.1.1, cleaned up optional dependencies, added test and docs groups
- [x] **Phase 1: Migrate to src layout structure** - Moved package to modern src/omero_annotate_ai/ structure, updated pyproject.toml configuration, verified all imports work correctly
- [x] **Phase 1: Organize omero_functions.py** - Analysis shows current organization is optimal (no changes needed)
- [x] **Phase 1: Update imports after code reorganization** - All imports verified working correctly

## Phase 2: Robust CI/CD Pipeline (2-3 days)

### ⏳ Pending
- [ ] **Phase 2: Create GitHub Actions workflow for multi-platform CI/CD**
- [ ] **Phase 2: Configure test matrix for Python 3.8-3.12 on Ubuntu/Windows/macOS**

## Phase 3: Automatic API Documentation (1-2 days)

### ⏳ Pending
- [ ] **Phase 3: Set up Sphinx for automatic API documentation generation**
- [ ] **Phase 3: Configure GitHub Pages deployment for documentation**

## Phase 4: Enhanced Testing (1-2 days)

### ⏳ Pending
- [ ] **Phase 4: Expand test suite with missing coverage areas**
- [ ] **Phase 4: Add installation and widget functionality tests**

## Notes

- Focus on keeping complexity manageable while ensuring professional-grade reliability
- Priority on your core requirements: API docs, multi-Python testing, multi-OS compatibility, CI/CD, unit tests
- Clean code organization following CLAUDE.md architectural guidelines

## Next Session Goals

1. Continue with Phase 1: Reorganize omero_functions.py
2. Set up basic GitHub Actions workflow
3. Test that everything still works after changes

---
*Last updated: 2025-01-07*