# Contributing

Thanks for contributing to Multi-Agent Lab.

## Development setup

Use Python 3.11 or newer. Create a virtual environment and install the package
with both provider adapters:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[anthropic,openai]'
```

The test suite is offline and does not require provider API keys:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -W error::ResourceWarning -m unittest discover -s tests -v
```

## Pull requests

- Keep changes focused and add tests for changed behavior.
- Never commit API keys, `.env` files, run databases, or generated artifacts.
- Update the README when CLI behavior or configuration changes.
- Open a pull request against `main`; both supported-Python CI jobs must pass.

For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
