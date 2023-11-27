import nltk
import ssl

# Alternative installer for: python -m nltk.downloader all
# If this error is obtained: "Error loading all: <urlopen error [SSL:
# Source: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')