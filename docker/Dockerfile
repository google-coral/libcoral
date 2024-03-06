ARG IMAGE
FROM ${IMAGE}

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN echo 'APT::Immediate-Configure false;' >> /etc/apt/apt.conf

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libc6-dev:arm64 \
      libc6-dev:armhf \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      sudo \
      debhelper \
      python3-all \
      python3-numpy \
      python3-setuptools \
      python3-six \
      python3-wheel \
      libpython3-dev \
      libpython3-dev:armhf \
      libpython3-dev:arm64 \
      build-essential \
      crossbuild-essential-armhf \
      crossbuild-essential-arm64 \
      libusb-1.0-0-dev \
      libusb-1.0-0-dev:arm64 \
      libusb-1.0-0-dev:armhf \
      zlib1g-dev \
      zlib1g-dev:armhf \
      zlib1g-dev:arm64 \
      pkg-config \
      p7zip-full \
      zip \
      unzip \
      curl \
      wget \
      git \
      vim \
      mc \
      software-properties-common

# Bionic Beaver == Ubuntu 18.04
RUN if grep 'Bionic Beaver' /etc/os-release > /dev/null; then \
      add-apt-repository ppa:ubuntu-toolchain-r/test \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-9 g++-9; \
    fi

# On older Ubuntu these packages can't be installed in a multi-arch fashion.
# Instead we download the debs and extract them for build time linking.
RUN mkdir /debs && chmod a=rwx /debs && cd /debs && apt-get update && apt-get download \
  libglib2.0-0 \
  libglib2.0-0:armhf \
  libglib2.0-0:arm64 \
  libglib2.0-dev \
  libglib2.0-dev:armhf \
  libglib2.0-dev:arm64 \
  libgstreamer1.0-0 \
  libgstreamer1.0-0:armhf \
  libgstreamer1.0-0:arm64 \
  libgstreamer1.0-dev \
  libgstreamer1.0-dev:armhf \
  libgstreamer1.0-dev:arm64 \
  libgstreamer-plugins-base1.0-0 \
  libgstreamer-plugins-base1.0-0:armhf \
  libgstreamer-plugins-base1.0-0:arm64 \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-base1.0-dev:armhf \
  libgstreamer-plugins-base1.0-dev:arm64

RUN for d in /debs/*.deb; do dpkg -x $d /usr/system_libs; done

RUN git clone https://github.com/raspberrypi/tools.git && \
    cd tools && \
    git reset --hard 4a335520900ce55e251ac4f420f52bf0b2ab6b1f

ARG BAZEL_VERSION=6.5.0
RUN wget -O /bazel https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    bash /bazel && \
    rm -f /bazel
