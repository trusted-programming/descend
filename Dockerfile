# syntax=docker/dockerfile:1.4
FROM ubuntu:plucky AS base

# ------------------------
# 1️⃣ Install dependencies
# ------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
    software-properties-common build-essential \
    llvm-20 llvm-20-dev clang-20 clang-tools-20 libclang-20-dev \
    lld-20 lldb-20 mlir-20-tools libmlir-20-dev cmake ninja-build \
    git python3 python3-dev pkg-config zlib1g-dev libedit-dev \
    libxml2-dev libzstd-dev libomp-20-dev libssl-dev \
    libncurses5-dev libgdbm-dev libnss3-dev libreadline-dev \
    libffi-dev curl pyenv libpolly-20-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ------------------------
# 2️⃣ Install pyenv and Python 3.11
# ------------------------
ENV PYENV_ROOT=/root/.pyenv
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
RUN pyenv install 3.11.9 && pyenv global 3.11.9

# ------------------------
# 3️⃣ Install Ascend toolkit silently
# ------------------------
COPY Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run /root
RUN chmod +x /root/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run && \
    /root/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run --full --quiet && \
    rm /root/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run

# ------------------------
# 4️⃣ Set up entrypoint script for runtime environment setup
# ------------------------
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ------------------------
# 5️⃣  Rust
# ------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rust_script.sh \
 && sh rust_script.sh -y --default-toolchain nightly-2025-08-04 \
 && rm rust_script.sh

# ------------------------
# 6 Python packages
# ------------------------
RUN pip install --upgrade pip \
 && pip install numpy==1.26.4 \
	decorator \
	sympy \
	scipy \
	attr \
	tornado \
	attrs \
	psutil \
	tornado

# ------------------------
# 7 Final cleanup
# ------------------------
RUN rm -rf /root/.cache /tmp/*

COPY ascend-rs /root/ascend-rs
RUN bash -c "cd /root/ascend-rs && source docker-entrypoint.sh && /root/.cargo/bin/cargo build"

WORKDIR /root
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
