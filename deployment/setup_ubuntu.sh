#!/bin/bash
# =============================================================================
#  SETUP SCRIPT — Ubuntu Environment for Artificial ECN Implementation
#  Run this FIRST on your Ubuntu VM: chmod +x setup_ubuntu.sh && sudo ./setup_ubuntu.sh
# =============================================================================
set -e

echo "============================================================"
echo "  Artificial ECN — Ubuntu Environment Setup"
echo "============================================================"

# --- Step 1: System Update ---
echo ""
echo ">>> Step 1: Updating system packages..."
apt-get update -y
apt-get upgrade -y

# --- Step 2: Install Mininet ---
echo ""
echo ">>> Step 2: Installing Mininet..."
apt-get install -y mininet

# Verify Mininet
echo "  Verifying Mininet installation..."
mn --version || { echo "ERROR: Mininet installation failed!"; exit 1; }
echo "  ✓ Mininet installed successfully"

# --- Step 3: Install supporting network tools ---
echo ""
echo ">>> Step 3: Installing network tools..."
apt-get install -y \
    iperf3 \
    iproute2 \
    tcpdump \
    net-tools \
    ethtool \
    openvswitch-switch

# Start Open vSwitch (needed by Mininet)
service openvswitch-switch start 2>/dev/null || true

# --- Step 4: Install Python and ML dependencies ---
echo ""
echo ">>> Step 4: Installing Python + ML libraries..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential

# Install ML libraries
pip3 install --upgrade pip
pip3 install \
    numpy \
    pandas \
    scikit-learn \
    lightgbm \
    xgboost \
    catboost \
    imbalanced-learn \
    joblib \
    matplotlib

echo "  ✓ Python ML libraries installed"

# --- Step 5: Enable ECN in kernel ---
echo ""
echo ">>> Step 5: Configuring kernel for ECN..."

# Enable ECN
sysctl -w net.ipv4.tcp_ecn=1

# Make it persistent across reboots
if ! grep -q "net.ipv4.tcp_ecn=1" /etc/sysctl.conf; then
    echo "net.ipv4.tcp_ecn=1" >> /etc/sysctl.conf
fi

# Also ensure TCP timestamps and SACK are enabled (needed for accurate measurements)
sysctl -w net.ipv4.tcp_timestamps=1
sysctl -w net.ipv4.tcp_sack=1

echo "  ✓ ECN enabled (net.ipv4.tcp_ecn=1)"

# --- Step 6: Verify everything ---
echo ""
echo "============================================================"
echo "  VERIFICATION"
echo "============================================================"

echo -n "  Mininet:    "; mn --version
echo -n "  Python:     "; python3 --version
echo -n "  iperf3:     "; iperf3 --version 2>&1 | head -1
echo -n "  LightGBM:   "; python3 -c "import lightgbm; print(lightgbm.__version__)"
echo -n "  XGBoost:    "; python3 -c "import xgboost; print(xgboost.__version__)"
echo -n "  ECN status: "; sysctl net.ipv4.tcp_ecn

echo ""
echo "============================================================"
echo "  ✓ SETUP COMPLETE!"
echo ""
echo "  Next steps:"
echo "  1. Transfer your trained model from Windows (upload to Drive)"
echo "  2. Run: python3 export_model.py  (on Windows first)"
echo "  3. Run: sudo python3 run_experiment.py  (on Ubuntu)"
echo "============================================================"
