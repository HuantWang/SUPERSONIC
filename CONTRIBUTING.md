# Contributing  <!-- omit in toc -->

**Table of Contents**

- [How to Contribute](#how-to-contribute)
- [Pull Requests](#pull-requests)
- [Code Style](#code-style)

---

## How to Contribute

We want to make contributing to SuperSonic as easy and transparent
as possible. The most helpful ways to contribute are:

1. Provide feedback.
   * [Report bugs](https://github.com/HuantWang/SUPERSONIC/issues). In
     particular, it’s important to report any crash or correctness bug. We use
     GitHub issues to track public bugs. Please ensure your description is clear
     and has sufficient instructions to be able to reproduce the issue.
   * Report issues when the documentation is incomplete or unclear, or an error
     message could be improved.
   * Make feature requests. Let us know if you have a use case that is not well
     supported, including as much detail as possible.
1. Contribute to the SuperSonic ecosystem.

## Pull Requests

We actively welcome your pull requests.

1. Fork [the repo](https://github.com/HuantWang/SUPERSONIC) and create
   your branch from `development`.
2. Follow the instructions for
   [building from source](https://github.com/HuantWang/SUPERSONIC/blob/master/INSTALL.md)
   to set up your environment.
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the [documentation](/docs/source).
5. Ensure the `make test` suite passes.
6. Make sure your code lints (see [Code Style](#code-style) below).
7. If you haven't already, complete the [Contributor License Agreement
   ("CLA")](#contributor-license-agreement-cla).

   
## Code Style

We want to ease the burden of code formatting using tools. Our code style
is simple:

* Python:
  [black](https://github.com/psf/black/blob/master/docs/the_black_code_style.md)
  and [isort](https://pypi.org/project/isort/).
* C++: [Google C++
  style](https://google.github.io/styleguide/cppguide.html) with 100
  character line length and `camelCaseFunctionNames()`.

We use [pre-commit](https://pre-commit.com/) to ensure that code is formatted
prior to committing. Before submitting pull requests, please run pre-commit. See
the [config file](/.pre-commit-config.yaml) for installation and usage
instructions.

Other common sense rules we encourage are:

* Prefer descriptive names over short ones.
* Split complex code into small units.
* When writing new features, add tests.
* Make tests deterministic.
* Prefer easy-to-use code over easy-to-read, and easy-to-read code over
  easy-to-write.
