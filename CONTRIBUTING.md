# Contributing

Read the [developer guide](docs/MAINTAINER_GUIDE.md) and [file map](docs/FILE_MAP.md), then create a branch for the change, for example `git switch -c fix/document-retry`.

Keep each change focused. Add a regression test when fixing behavior, run the relevant checks, and describe the problem and result in the pull request. Use invented data in tests. Keep personal files, credentials, recordings and downloaded models out of Git.

Test changes on the target operating system. Documentation-only changes need link and content checks; they don't require another model download. Keep measured results separate from simulated tests and record remaining limits in [test results](docs/CURRENT_STATE.md).

Preserve the license notices and the user's saved model, voice and data when changing setup.
