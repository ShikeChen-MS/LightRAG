# building stage
# build python and install dependencies
FROM mcr.microsoft.com/azurelinux/base/core:3.0 AS builder

ARG PYTHON_VERSION=3.13.2
ARG PYTHON_CMD=python3.13

RUN tdnf check-update && \
    tdnf update -y && \
    tdnf clean all -y

# Update the package list and install build dependencies
RUN tdnf install -y \
    wget \
    make \
    gcc \
    openssl-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    binutils \
    glibc-devel \
    kernel-headers \
    awk \
    ca-certificates \
    tar

# Download and install Python 3.12
RUN cd / && \
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

RUN tar xzf Python-$PYTHON_VERSION.tgz

RUN cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations

RUN cd Python-$PYTHON_VERSION && \
    make altinstall

RUN cd / && \
    $PYTHON_CMD -m pip install -U pip

RUN rm -f Python-$PYTHON_VERSION.tgz

COPY requirements.txt .

# install pip dependencies
RUN python3.12 -m pip install --user --no-cache-dir -r requirements.txt


# Final stage
FROM mcr.microsoft.com/azurelinux/base/core:3.0

WORKDIR /app

RUN tdnf check-update
RUN tdnf update -y
RUN tdnf clean all

# Copy Python installation from the builder stage
COPY --from=builder /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY ./lightrag ./lightrag
COPY setup.py .

RUN python3.12 -m pip install .
# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose the default port
EXPOSE 9621

# Set entrypoint
ENTRYPOINT ["python3.12", "-m", "lightrag.api.lightrag_server"]
