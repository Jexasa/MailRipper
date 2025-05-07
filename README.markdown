# MailRipper

**MailRipper** is a stealthy OSINT tool for extracting email usernames from a domain using web scraping, Hunter.io API queries, and email pattern generation.

``` 
 Version: 1.2.0
 Date: 05/07/25
 Author: KitsiosM (ksexasa)
```

## Features

- Scrapes websites as stealthy as possible (user-agent rotation, delays, proxy support).
- Queries Hunter.io API for verified emails and patterns.
- Generates emails from names with MX record validation.
- Supports txt, JSON, CSV output formats.
- Configurable via `config.ini` or environment variables.
- Structured logging with file output option.
- Proxy support, including Tor for anonymous requests.

## Prerequisites

- Python 3.6+
- [Hunter.io API key](https://hunter.io/api)
- Dependencies: `aiohttp`, `aiohttp_socks`, `beautifulsoup4`, `dnspython`, `structlog`, `tenacity`, `tldextract`, `tqdm`
- (Optional) Tor service for proxy (e.g., `socks5://localhost:9050`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jexasa/mail-ripper.git
   cd mail-ripper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Tor for proxy:
   - Install Tor: `sudo apt install tor` (Linux) or equivalent.
   - Start Tor service: `tor` (runs on `socks5://localhost:9050` by default).
   - Verify Tor is running: `curl --socks5 localhost:9050 https://check.torproject.org`.

## Usage

Run with a domain and API key. Optionally, provide a names file, output format, or proxy.

### Basic Run
```bash
python3 mail_ripper.py example.com --api-key YOUR_API_KEY
```
Saves emails to `emails.txt`.

### With Names and JSON Output
Use a `names.txt` file (format: `First Last` per line):
```bash
python3 mail_ripper.py example.com --api-key YOUR_API_KEY --names names.txt --format json --output results.json
```

### Using Tor Proxy
```bash
python3 mail_ripper.py example.com --api-key YOUR_API_KEY --proxy socks5://localhost:9050
```

### Debug Mode
```bash
python3 mail_ripper.py example.com --api-key YOUR_API_KEY --log-level DEBUG --log-file mail_ripper.log
```

See all options:
```bash
python3 mail_ripper.py --help
```

## Configuration

Create `config.ini` for defaults (optional):
```ini
[DEFAULT]
api_key = YOUR_API_KEY
output = emails.txt
log_level = INFO
log_file = mail_ripper.log
proxy = socks5://localhost:9050
```

Or set environment variables:
```bash
export MAIL_RIPPER_API_KEY=YOUR_API_KEY
export MAIL_RIPPER_PROXY=socks5://localhost:9050
```

## Contributing

Fork, create a feature branch, commit changes, and open a pull request. Follow the contribution guidelines in `CONTRIBUTING.md`.

## License

MIT License. See `LICENSE` for details.

---

Stealthy email extraction made simple! 🚀
