# -----------------------------------------------------------------------------
# Agent Template with License Attribution
# Provides license attribution for third-party software dependencies
# -----------------------------------------------------------------------------

# Build args for version injection
ARG VERSION=unknown
ARG BUILD_TIMESTAMP=unknown
ARG GIT_COMMIT=unknown

# -----------------------------------------------------------------------------
# Builder stage – install deps, build app, and collect ALL licenses
# -----------------------------------------------------------------------------
FROM python:3.14-slim AS builder

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt* ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Copy source code
COPY . .

# ---- LICENSE COLLECTION ----
RUN mkdir -p /app/licenses/python /app/licenses/debian /app/licenses/python_notices

# 1) Install license collection tool
RUN pip install --no-cache-dir pip-licenses

# 2) Capture ALL Python dependencies with license texts
RUN pip freeze --exclude-editable > /app/licenses/python/THIRD_PARTY_REQUIREMENTS.txt \
    && pip-licenses \
        --format=json \
        --with-authors \
        --with-urls \
        --with-license-file \
        --no-license-path \
        > /app/licenses/python/THIRD_PARTY_LICENSES.json

# 3) Generate human-readable attribution file with embedded license texts
RUN python - <<'PY'
import json, os
p = "/app/licenses/python/THIRD_PARTY_LICENSES.json"
data = json.load(open(p))
out = "/app/licenses/python/ATTRIBUTIONS.md"
with open(out, "w", encoding="utf-8") as f:
    f.write("# Third-Party Python Packages\n\n")
    f.write("This agent includes the following third-party Python packages:\n\n")
    for row in sorted(data, key=lambda r: r["Name"].lower()):
        f.write(f"## {row.get('Name','')} {row.get('Version','')}\n")
        f.write(f"- License: {row.get('License','Unknown')}\n")
        if row.get("URL"): f.write(f"- URL: {row['URL']}\n")
        if row.get("Author"): f.write(f"- Author: {row['Author']}\n")
        txt = row.get("LicenseText")
        if txt and len(txt) < 50000:  # Skip extremely long licenses
            f.write("\n<details><summary>License text</summary>\n\n")
            f.write("```\n")
            f.write(txt.strip())
            f.write("\n```\n")
            f.write("</details>\n")
        f.write("\n")
print("✅ Generated Python attribution file")
PY

# 4) Export dependency tree for reproducibility
RUN pip list --format=json > /app/licenses/python/DEPENDENCY_LIST.json \
    && echo "# Python Dependencies for Agent\n" > /app/licenses/python/DEPENDENCY_TREE.md \
    && echo "## Production Dependencies\n" >> /app/licenses/python/DEPENDENCY_TREE.md \
    && cat requirements.txt | grep -v '^#' | grep -v '^$' | while read dep; do \
        echo "- $dep" >> /app/licenses/python/DEPENDENCY_TREE.md; \
    done

# 5) Copy NOTICE files from Python packages
RUN python - <<'PY'
import sys, pathlib, shutil
dest = pathlib.Path("/app/licenses/python_notices")
dest.mkdir(parents=True, exist_ok=True)
for p in map(pathlib.Path, sys.path):
    if p.exists() and "site-packages" in str(p):
        for item in p.iterdir():
            if item.is_dir():
                for name in ("NOTICE", "NOTICE.txt", "NOTICE.md", "NOTICE.rst"):
                    n = item / name
                    if n.exists():
                        shutil.copy2(n, dest / f"{item.name}-{name}")
print("✅ Collected Python NOTICE files")
PY

# 6) Capture Debian/Ubuntu system packages
RUN dpkg-query -W -f='${Package} ${Version} ${Maintainer}\n' > /app/licenses/debian/INSTALLED_PACKAGES.txt \
    && mkdir -p /app/licenses/debian/copyrights \
    && for pkg in $(dpkg-query -W -f='${Package}\n'); do \
        src="/usr/share/doc/$pkg/copyright"; \
        if [ -f "$src" ]; then \
            cp "$src" "/app/licenses/debian/copyrights/${pkg}-copyright"; \
        fi; \
    done

# 7) Create build environment attribution
RUN echo "# Agent Build Environment Attribution\n" > /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "## Build Tools Used\n" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "- Python: $(python --version) (PSF License)" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "- pip: $(pip --version | cut -d' ' -f2) (MIT License)" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "- Base OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"') (Various Licenses)" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "\n## Build Information\n" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "- Build date: $(date)" >> /app/licenses/BUILD_ENVIRONMENT.md \
    && echo "- Platform: $(uname -m)" >> /app/licenses/BUILD_ENVIRONMENT.md

# 8) Run license verification during build
RUN python - <<'PY'
import json, os, sys

print("🔍 Verifying license attribution coverage...\n")

try:
    # Read expected dependencies from requirements.txt
    expected_deps = {}
    requirements_file = "/app/requirements.txt"
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                    expected_deps[pkg_name.lower()] = line

    # Read generated licenses
    with open('/app/licenses/python/THIRD_PARTY_LICENSES.json', 'r') as f:
        licenses = json.load(f)

    # Check coverage
    attributed_packages = set(pkg.get('Name', '').lower() for pkg in licenses)

    print(f"📊 Attribution Statistics:")
    print(f"   - Total attributed packages: {len(licenses)}")
    print(f"   - Expected production deps: {len(expected_deps)}")

    missing = []
    found = []
    for dep_name in expected_deps.keys():
        if dep_name.replace('_', '-') in attributed_packages or dep_name.replace('-', '_') in attributed_packages:
            found.append(dep_name)
            print(f"✅ {dep_name}")
        else:
            missing.append(dep_name)
            print(f"❌ {dep_name} - NOT FOUND")

    print(f"\n📋 Summary: Found {len(found)}/{len(expected_deps)}, Missing: {len(missing)}")

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")

    # Check for problematic licenses
    license_types = sorted(set(pkg.get('License', 'Unknown') for pkg in licenses))
    problematic = [lt for lt in license_types if lt and any(term in lt.lower() for term in ['gpl', 'agpl', 'copyleft'])]

    if problematic:
        print(f"\n⚠️  WARNING: Found potentially copyleft licenses: {', '.join(problematic)}")

    print(f"\n📄 License types: {', '.join(license_types)}")

    if len(missing) == 0:
        print("\n🎉 SUCCESS: All dependencies properly attributed!")
    else:
        print(f"\n❌ INCOMPLETE: {len(missing)} dependencies missing attribution")
        # Don't fail build for missing deps, just warn

except Exception as e:
    print(f"❌ Attribution verification failed: {e}")
    sys.exit(1)
PY

# 9) Generate verification checksums
RUN find /app/licenses -name "*.json" -o -name "*.txt" -o -name "*.md" | \
    sort | xargs sha256sum > /app/licenses/ATTRIBUTION_CHECKSUMS.txt

# 10) Create license summary
RUN echo "# Agent Container - License Attribution\n" > /app/licenses/README.md \
    && echo "This container includes license attribution for third-party dependencies.\n" >> /app/licenses/README.md \
    && echo "## Coverage Areas\n" >> /app/licenses/README.md \
    && echo "- **Python Dependencies**: Runtime packages with license information\n" >> /app/licenses/README.md \
    && echo "- **Build Environment**: Python runtime, pip, Debian/Ubuntu base\n" >> /app/licenses/README.md \
    && echo "- **System Packages**: Debian/Ubuntu package information\n" >> /app/licenses/README.md \
    && echo "- **Integrity Verification**: SHA256 checksums of attribution files\n" >> /app/licenses/README.md \
    && echo "\n## Quick Access\n" >> /app/licenses/README.md \
    && echo "- Python attributions: /app/licenses/python/ATTRIBUTIONS.md\n" >> /app/licenses/README.md \
    && echo "- Debian attributions: /app/licenses/debian/INSTALLED_PACKAGES.txt\n" >> /app/licenses/README.md \
    && echo "- Build environment: /app/licenses/BUILD_ENVIRONMENT.md\n" >> /app/licenses/README.md \
    && echo "- Verification checksums: /app/licenses/ATTRIBUTION_CHECKSUMS.txt\n" >> /app/licenses/README.md

# 11) Remove license scanning tools to keep runtime clean
RUN pip uninstall -y pip-licenses || true

# -----------------------------------------------------------------------------
# Final stage – minimal runtime image with attribution
# -----------------------------------------------------------------------------
FROM python:3.14-slim

# Re-declare args for final stage
ARG VERSION=unknown
ARG BUILD_TIMESTAMP=unknown
ARG GIT_COMMIT=unknown

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install minimal runtime OS dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        # Add any runtime OS deps here if needed for your specific agent
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages and application from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/*.py /app/
COPY --from=builder /app/agent.yaml /app/
COPY --from=builder /app/metadata.yaml /app/
COPY --from=builder /app/requirements.txt /app/

# Copy project LICENSE
COPY LICENSE /app/LICENSE

# Copy license attribution artifacts
COPY --from=builder /app/licenses /app/licenses

# Create version file for runtime access
RUN echo "{\"version\": \"${VERSION}\", \"build_timestamp\": \"${BUILD_TIMESTAMP}\", \"git_commit\": \"${GIT_COMMIT}\"}" > /app/version.json

# OCI labels for container metadata
# TODO: Customize title and vendor for your agent
LABEL org.opencontainers.image.title="Your Agent Name" \
      org.opencontainers.image.description="AI agent with license attribution" \
      org.opencontainers.image.vendor="Your Organization" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.license.files="/app/licenses" \
      org.opencontainers.image.license.verification="/app/licenses/ATTRIBUTION_CHECKSUMS.txt" \
      org.opencontainers.image.source="https://github.com/agentsystems/agent-template" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_TIMESTAMP}" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

# Create non-root user for security
RUN useradd -u 1001 -m appuser
USER 1001

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
