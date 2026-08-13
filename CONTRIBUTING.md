## Contributing

The best way to contribute to the MLCommons is to get involved with one of our many project communities. You can find more information about getting involved with MLCommons [here](https://mlcommons.org/en/get-involved/#getting-started). 

Generally we encourage people to become a MLCommons member if they wish to contribute to MLCommons projects, but outside pull requests are very welcome too.

To get started contributing code, you or your organization needs to sign the MLCommons CLA found at the [MLC policies page](https://mlcommons.org/en/policies/). Once you or your organization has signed the corporate CLA, please fill out this [CLA sign up form](https://forms.gle/Ew1KkBVpyeJDuRw67) form to get your specific GitHub handle authorized so that you can start contributing code under the proper license.

MLCommons project work is tracked with issue trackers and pull requests. Modify the project in your own fork and issue a pull request once you want other developers to take a look at what you have done and discuss the proposed changes. Ensure that cla-bot and other checks pass for your Pull requests.

## Pull Request Templates

This repository uses PR templates plus a GitHub Action so the right checklist is added whether the PR is opened from the GitHub UI, `gh` / the API, or a fork.

Add one or both labels; the [Add PR checklist](.github/workflows/pr-checklist.yml) workflow appends the matching checklist to the PR description if it is not already there:

| Label | Checklist | When to use |
|-------|-----------|-------------|
| `new-benchmark` | [`.github/PULL_REQUEST_TEMPLATE/new_benchmark.md`](.github/PULL_REQUEST_TEMPLATE/new_benchmark.md) | Introducing a new benchmark to MLPerf Training. Includes the full checklist from [issue #810](https://github.com/mlcommons/training/issues/810). |
| `new-round` | [`.github/PULL_REQUEST_TEMPLATE/new_round.md`](.github/PULL_REQUEST_TEMPLATE/new_round.md) | Preparing and introducing a new training round. |

You can also pick a template in the GitHub UI, pass `?template=<file>.md` in the PR URL, or run `gh pr create --template <file>.md --label <label>`.

Please complete all applicable checklist items before requesting review. If a checklist item does not apply to your PR, mark it as complete and note why it was skipped.
