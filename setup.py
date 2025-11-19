from setuptools import setup

setup(
    name="tiktok-collector",
    version="1.5.0",
    py_modules=["tiktok_tool"],
    description=(
        "TikTok Collector — batch-friendly CLI for scraping TikTok links, "
        "fetching metadata, and optionally downloading videos "
        "using a single Playwright session."
    ),
    long_description=(
        "TikTok Collector is a command-line tool that scrapes TikTok video URLs "
        "for hashtags, users, or keyword searches, fetches video metadata, and "
        "optionally downloads videos. It reuses a single Playwright browser "
        "session to minimize CAPTCHA friction and supports CSV + SQLite outputs."
    ),
    long_description_content_type="text/plain",
    author="Qasem Exirifard",
    author_email="qasem.exirifard@international.gc.ca",
    url="..",  # replace with your repo if you like
    license="GAC",  # change this if you prefer another license
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv",
        "tqdm",
        "yt-dlp",
        "requests",
        "pandas",
        "playwright",
    ],
    entry_points={
        "console_scripts": [
            # installs a `tiktok-tool` command that runs main()
            "tiktok-tool = tiktok_tool:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Multimedia :: Video",
        "Topic :: Utilities",
    ],
)
