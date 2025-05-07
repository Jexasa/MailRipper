#!/usr/bin/env python3

import argparse
import asyncio
import configparser
import csv
import json
import logging
import os
import random
import re
import sys
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
import dns.resolver
import structlog
import tldextract 
from aiohttp_socks import ProxyConnector
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Constants
HUNTER_API_URL = "https://api.hunter.io/v2/domain-search"
EMAIL_PATTERNS = [
    "{first}.{last}",
    "{first}{last}",
    "{first}_{last}",
    "{last}.{first}",
    "{f}{last}",
    "{first}{l}",
    "{first}",
    "{last}"
]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]
CONFIG_FILE = "config.ini"

# ASCII Logo
LOGO = """
+---+---+---+ 
| M | R | P | 
+---+---+---+ 
 MailRipper
 Version: 1.2.0
 Date: 05/07/25
 Writer: KitsiosM (ksexasa)
 Purpose: Stealthy OSINT Email Extraction 
"""

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Configure logging with file output if specified.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file (Optional[str]): Path to log file, if any.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )

def load_config() -> configparser.ConfigParser:
    """Load configuration from config.ini if it exists.

    Returns:
        configparser.ConfigParser: Configuration object with defaults.
    """
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'api_key': '',
        'output': 'emails.txt',
        'log_level': 'INFO',
        'log_file': '',
        'proxy': ''
    }
    if Path(CONFIG_FILE).exists():
        config.read(CONFIG_FILE)
    return config

def validate_domain(domain: str) -> str:
    """Validate and extract the registered domain.

    Args:
        domain (str): The domain to validate (e.g., example.com).

    Returns:
        str: The registered domain (e.g., example.com).

    Raises:
        ValueError: If the domain is invalid.
    """
    ext = tldextract.extract(domain)
    registered_domain = ext.registered_domain
    if not registered_domain:
        raise ValueError(f"Invalid domain: {domain}")
    logger.info("Validated domain", domain=domain, registered_domain=registered_domain)
    return registered_domain

def validate_email(email: str) -> bool:
    """Validate an email address by checking MX records.

    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email's domain has valid MX records, False otherwise.
    """
    domain = email.split('@')[1]
    try:
        dns.resolver.resolve(domain, 'MX')
        return True
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout):
        logger.warning("Invalid email domain", email=email, domain=domain)
        return False

def validate_proxy(proxy: str) -> bool:
    """Validate the proxy URL format.

    Args:
        proxy (str): The proxy URL (e.g., socks5://localhost:9050).

    Returns:
        bool: True if valid, False otherwise.
    """
    if not proxy:
        return True
    try:
        parsed = urlparse(proxy)
        if parsed.scheme not in ('http', 'https', 'socks5', 'socks4') or not parsed.netloc:
            return False
        return True
    except ValueError:
        logger.error("Invalid proxy format", proxy=proxy)
        return False

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text using regex.

    Args:
        text (str): The text to search for emails.

    Returns:
        List[str]: A list of unique email addresses found.
    """
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-0-9.-]+\.[a-zA-Z]{2,}"
    emails = list(set(re.findall(email_regex, text)))
    logger.debug("Extracted emails", count=len(emails), emails=emails)
    return emails

async def scrape_website(domain: str, session: aiohttp.ClientSession, proxy: Optional[str] = None) -> List[str]:
    """Scrape the website for email addresses from multiple pages.

    Args:
        domain (str): The domain to scrape.
        session (aiohttp.ClientSession): The async HTTP session.
        proxy (Optional[str]): Proxy URL for requests, if any.

    Returns:
        List[str]: A list of unique email addresses found.
    """
    urls = [
        f"https://{domain}",
        f"https://{domain}/contact",
        f"https://{domain}/about"
    ]
    emails = []
    headers = {"User-Agent": random.choice(USER_AGENTS)}

    for url in tqdm(urls, desc="Scraping pages", leave=False):
        try:
            async with session.get(url, headers=headers, timeout=10, proxy=proxy) as response:
                if response.status != 200:
                    logger.warning("Non-200 response", url=url, status=response.status)
                    continue
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                emails.extend(extract_emails(soup.get_text()))
                # Extract additional links to follow (basic crawling)
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    full_url = urljoin(url, href)
                    if domain in full_url and full_url not in urls and len(urls) < 10:
                        urls.append(full_url)
            await asyncio.sleep(random.uniform(1, 3))  # Random delay for stealth
        except aiohttp.ClientError as e:
            logger.warning("Failed to scrape URL", url=url, error=str(e))
        except Exception as e:
            logger.error("Unexpected error during scraping", url=url, error=str(e))
    
    valid_emails = [email for email in set(emails) if validate_email(email)]
    logger.info("Completed web scraping", domain=domain, email_count=len(valid_emails), proxy=proxy)
    return valid_emails

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def hunter_search(domain: str, api_key: str, session: aiohttp.ClientSession, proxy: Optional[str] = None) -> Dict[str, any]:
    """Query Hunter.io API for email addresses and pattern.

    Args:
        domain (str): The domain to query.
        api_key (str): The Hunter.io API key.
        session (aiohttp.ClientSession): The async HTTP session.
        proxy (Optional[str]): Proxy URL for requests, if any.

    Returns:
        Dict[str, any]: A dictionary with emails and pattern.
    """
    params = {"domain": domain, "api_key": api_key}
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        async with session.get(HUNTER_API_URL, params=params, headers=headers, timeout=10, proxy=proxy) as response:
            response.raise_for_status()
            data = await response.json()
            emails = [email["value"] for email in data["data"].get("emails", [])]
            pattern = data["data"].get("pattern") or EMAIL_PATTERNS[0]
            logger.info("Hunter.io query successful", domain=domain, email_count=len(emails), proxy=proxy)
            return {"emails": emails, "pattern": pattern}
    except aiohttp.ClientError as e:
        logger.warning("Hunter.io query failed", domain=domain, error=str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error in Hunter.io query", domain=domain, error=str(e))
        return {"emails": [], "pattern": EMAIL_PATTERNS[0]}

def load_names(names_file: Optional[str]) -> List[Tuple[str, str]]:
    """Load names from a file.

    Args:
        names_file (Optional[str]): Path to the names file.

    Returns:
        List[Tuple[str, str]]: A list of (first, last) name tuples.

    Raises:
        SystemExit: If the file is not found or cannot be read.
    """
    names = []
    if names_file:
        try:
            with Path(names_file).open("r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        names.append((parts[0], parts[-1]))
            logger.info("Loaded names", file=names_file, count=len(names))
        except FileNotFoundError:
            logger.error("Names file not found", file=names_file)
            sys.exit(1)
        except Exception as e:
            logger.error("Error reading names file", file=names_file, error=str(e))
            sys.exit(1)
    return names

def generate_emails(names: List[Tuple[str, str]], domain: str, pattern: str) -> List[str]:
    """Generate email permutations based on names and pattern.

    Args:
        names (List[Tuple[str, str]]): List of (first, last) name tuples.
        domain (str): The domain for email generation.
        pattern (str): The email pattern to use.

    Returns:
        List[str]: A list of generated email addresses.
    """
    emails = []
    for first, last in tqdm(names, desc="Generating emails", leave=False):
        try:
            email = f"{pattern.format(first=first.lower(), last=last.lower(), f=first[0], l=last[0])}@{domain}"
            if validate_email(email):
                emails.append(email)
        except KeyError as e:
            logger.warning("Invalid pattern", pattern=pattern, name=f"{first} {last}", error=str(e))
    valid_emails = list(set(emails))
    logger.info("Generated emails", domain=domain, count=len(valid_emails))
    return valid_emails

def save_results(emails: List[str], output_file: str, output_format: str) -> None:
    """Save email results to a file in the specified format.

    Args:
        emails (List[str]): List of email addresses to save.
        output_file (str): Path to the output file.
        output_format (str): Format of the output ('txt', 'json', 'csv').

    Raises:
        SystemExit: If the file cannot be written.
    """
    try:
        with Path(output_file).open("w") as f:
            if output_format == "json":
                json.dump({"emails": emails}, f, indent=2)
            elif output_format == "csv":
                writer = csv.writer(f)
                writer.writerow(["Email"])
                for email in emails:
                    writer.writerow([email])
            else:  # txt
                for email in emails:
                    f.write(f"{email}\n")
        logger.info("Saved results", file=output_file, format=output_format, count=len(emails))
    except IOError as e:
        logger.error("Failed to write output file", file=output_file, error=str(e))
        sys.exit(1)

def validate_api_key(api_key: str) -> bool:
    """Validate the Hunter.io API key format.

    Args:
        api_key (str): The API key to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    # Hunter.io API keys are typically 40 characters long
    if len(api_key) == 40 and api_key.isalnum():
        return True
    logger.error("Invalid API key format", key_length=len(api_key))
    return False

async def main():
    """Main function to orchestrate email discovery."""
    # Help text with sample commands
    sample_commands = """
Examples:
  Basic run:
    python3 mail_ripper.py example.com --api-key YOUR_API_KEY

  With names file and JSON output:
    python3 mail_ripper.py example.com --api-key YOUR_API_KEY --names names.txt --format json --output results.json

  Using Tor proxy:
    python3 mail_ripper.py example.com --api-key YOUR_API_KEY --proxy socks5://localhost:9050

  Debug mode with log file:
    python3 mail_ripper.py example.com --api-key YOUR_API_KEY --log-level DEBUG --log-file mail_ripper.log
    """

    parser = argparse.ArgumentParser(
        description="Stealthy OSINT email extraction tool.",
        epilog=sample_commands,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("domain", help="Target domain (e.g., example.com)")
    parser.add_argument("--api-key", help="Hunter.io API key (or set MAIL_RIPPER_API_KEY env var)")
    parser.add_argument("--names", help="File with employee names (format: first last, one per line)")
    parser.add_argument("--output", default="emails.txt", help="Output file for results (default: emails.txt)")
    parser.add_argument("--format", choices=["txt", "json", "csv"], default="txt",
                        help="Output format (default: txt)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging level (default: INFO)")
    parser.add_argument("--log-file", help="Log output to a file")
    parser.add_argument("--proxy", help="Proxy URL (e.g., socks5://localhost:9050 for Tor)")
    args = parser.parse_args()

    # Display logo
    print(LOGO)

    # Load config
    config = load_config()
    api_key = args.api_key or os.getenv("MAIL_RIPPER_API_KEY") or config['DEFAULT']['api_key']
    output_file = args.output or config['DEFAULT']['output']
    log_level = args.log_level or config['DEFAULT']['log_level']
    log_file = args.log_file or config['DEFAULT'].get('log_file')
    proxy = args.proxy or os.getenv("MAIL_RIPPER_PROXY") or config['DEFAULT'].get('proxy')

    # Setup logging
    setup_logging(log_level, log_file)

    # Validate inputs
    if not api_key:
        logger.error("Hunter.io API key is required")
        sys.exit(1)
    if not validate_api_key(api_key):
        sys.exit(1)
    if args.names and not Path(args.names).is_file():
        logger.error("Names file does not exist", file=args.names)
        sys.exit(1)
    if not validate_proxy(proxy):
        sys.exit(1)
    try:
        domain = validate_domain(args.domain)
    except ValueError as e:
        logger.error("Domain validation failed", error=str(e))
        sys.exit(1)

    # Load data
    names = load_names(args.names)
    logger.info("Starting MailRipper", domain=domain, names_count=len(names), proxy=proxy)

    # Configure session with proxy if provided
    connector = ProxyConnector.from_url(proxy) if proxy else None
    async with aiohttp.ClientSession(connector=connector) as session:
        hunter_task = asyncio.create_task(hunter_search(domain, api_key, session, proxy))
        web_task = asyncio.create_task(scrape_website(domain, session, proxy))
        hunter_data, web_emails = await asyncio.gather(hunter_task, web_task, return_exceptions=True)
        
        if isinstance(hunter_data, Exception):
            logger.error("Hunter.io task failed", error=str(hunter_data))
            hunter_data = {"emails": [], "pattern": EMAIL_PATTERNS[0]}
        if isinstance(web_emails, Exception):
            logger.error("Web scraping task failed", error=str(web_emails))
            web_emails = []

        all_emails = list(set(hunter_data["emails"] + web_emails))
        pattern = hunter_data["pattern"]
        logger.debug("Detected email pattern", pattern=pattern)

        # Generate emails if names provided
        generated_emails = generate_emails(names, domain, pattern) if names else []
        results = list(set(all_emails + generated_emails))

    # Save results
    if results:
        save_results(results, output_file, args.format)
    else:
        logger.warning("No emails found", domain=domain)

if __name__ == "__main__":
    asyncio.run(main())