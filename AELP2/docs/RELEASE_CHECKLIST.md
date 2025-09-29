# Release Checklist

- Verify all TODO items are Done in docs
- Run GX checks and Prefect flows in dry-run
- Confirm ops_flow_runs logs ok:true
- Tag version and update CHANGELOG.md
- Gate prod deploy behind GATES_ENABLED and ALLOW_* defaults 0
