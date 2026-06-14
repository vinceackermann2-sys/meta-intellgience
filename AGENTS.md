# AI Coding Agent Guidelines

This project has strict rules to prevent destructive behaviors, package-lock.json compilation failures, and accidental file deletion.

## 🛑 STRICT RULES AGAINST DESTRUCTIVE BEHAVIORS
1. **Never Delete Files in `/src`:** Under no circumstances should you delete, clear, or completely rewrite files in `/src/` or the root folder unless explicitly and unambiguously requested by the user.
2. **Never Perform Full Cleanups/Wipes:** Do not attempt to "start fresh" by deleting existing features, views, routes, or assets. Always modify files incrementally.
3. **Never Delete `package.json` Elements on a Whim:** If there is an installation issue, do not randomly wipe or strip out existing dependencies.

## 📦 SAFE DEPENDENCY & PACKAGE INSTALLATION PROTOCOLS
If a package installation or `npm install` command fails, follow these steps step-by-step:
1. **Check for Peer Dependency Conflicts:** If you see conflicts related to React versions (e.g., React 19 vs React 18 packages), do NOT delete files. Run installations with specific options or adjust packages to compatible versions.
2. **Use the Dedicated Installation Tool:** Always prefer using `install_applet_package` or `install_applet_dependencies` instead of writing custom script executions to modify lockfiles.
3. **Explain and Consult:** If a dependency conflict is unresolvable without major changes to `package.json`, immediately stop, explain the exact error mismatch, and ask the user how they would like to proceed. Never execute destructive file deletions or full directory wipes as automated recovery steps.
