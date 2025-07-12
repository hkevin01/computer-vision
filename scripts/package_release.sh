#!/bin/bash

# 🎯 PROFESSIONAL INSTALLER GENERATION SCRIPT
# Creates cross-platform installation packages for the Stereo Vision application

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
APP_NAME="StereoVision3D"
APP_VERSION="1.0.0"
APP_DESCRIPTION="High-Performance Stereo Vision 3D Point Cloud Generator"
APP_VENDOR="StereoVision Technologies"
APP_CONTACT="support@stereovision3d.com"
APP_URL="https://stereovision3d.com"

# Directories
BUILD_DIR="$PROJECT_ROOT/build"
PACKAGE_DIR="$PROJECT_ROOT/packages"
INSTALLER_DIR="$PROJECT_ROOT/installer"
RESOURCES_DIR="$PROJECT_ROOT/installer/resources"

# Parse command line arguments
PACKAGE_TYPE="all"
TARGET_ARCH="x64"
BUILD_TYPE="Release"

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            PACKAGE_TYPE="$2"
            shift 2
            ;;
        --arch)
            TARGET_ARCH="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --type TYPE      Package type: deb, rpm, msi, dmg, appimage, or all"
            echo "  --arch ARCH      Target architecture: x64, arm64"
            echo "  --build-type     Build type: Release, Debug"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
print_status "Detected OS: $OS"

# Create directory structure
setup_directories() {
    print_status "Setting up directory structure..."
    
    mkdir -p "$PACKAGE_DIR"
    mkdir -p "$INSTALLER_DIR"
    mkdir -p "$RESOURCES_DIR"
    mkdir -p "$INSTALLER_DIR/scripts"
    mkdir -p "$INSTALLER_DIR/templates"
    
    print_success "Directory structure created"
}

# Build the application
build_application() {
    print_status "Building application for packaging..."
    
    if [ ! -d "$BUILD_DIR" ]; then
        mkdir -p "$BUILD_DIR"
    fi
    
    cd "$BUILD_DIR"
    
    # Configure with install prefix
    cmake .. \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_INSTALL_PREFIX="/opt/$APP_NAME" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DPACKAGE_BUILD=ON
    
    # Build
    cmake --build . --config "$BUILD_TYPE" --parallel "$(nproc 2>/dev/null || echo 4)"
    
    # Install to staging directory
    DESTDIR="$INSTALLER_DIR/staging" cmake --install . --config "$BUILD_TYPE"
    
    cd "$PROJECT_ROOT"
    print_success "Application built and staged"
}

# Create desktop entry file
create_desktop_entry() {
    cat > "$RESOURCES_DIR/stereovision3d.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=$APP_NAME
Comment=$APP_DESCRIPTION
Exec=/opt/$APP_NAME/bin/stereo_vision_app
Icon=/opt/$APP_NAME/share/icons/stereovision3d.png
Terminal=false
Categories=Graphics;3DGraphics;Engineering;Science;
Keywords=stereo;vision;3D;point cloud;computer vision;
StartupNotify=true
MimeType=application/x-stereovision-project;
EOF
    print_success "Desktop entry created"
}

# Create application icon (placeholder)
create_application_icon() {
    # In a real implementation, this would be a proper icon file
    # For now, we'll create a placeholder
    mkdir -p "$RESOURCES_DIR/icons"
    
    # Create a simple SVG icon as placeholder
    cat > "$RESOURCES_DIR/icons/stereovision3d.svg" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<svg width="128" height="128" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2196F3;stop-opacity:1" />
    </linearGradient>
  </defs>
  <circle cx="64" cy="64" r="60" fill="url(#grad)" stroke="#1976D2" stroke-width="2"/>
  <text x="64" y="72" font-family="Arial" font-size="20" fill="white" text-anchor="middle">3D</text>
  <rect x="20" y="40" width="25" height="20" fill="white" opacity="0.8"/>
  <rect x="83" y="40" width="25" height="20" fill="white" opacity="0.8"/>
</svg>
EOF

    # Convert SVG to PNG using ImageMagick (if available)
    if command -v convert &> /dev/null; then
        convert "$RESOURCES_DIR/icons/stereovision3d.svg" \
                -resize 128x128 \
                "$RESOURCES_DIR/icons/stereovision3d.png"
    else
        # Create a simple placeholder PNG if ImageMagick is not available
        print_warning "ImageMagick not found, using placeholder icon"
        cp "$RESOURCES_DIR/icons/stereovision3d.svg" "$RESOURCES_DIR/icons/stereovision3d.png"
    fi
    
    print_success "Application icon created"
}

# Create Debian package
create_deb_package() {
    print_status "Creating Debian package..."
    
    DEB_DIR="$INSTALLER_DIR/deb"
    mkdir -p "$DEB_DIR/DEBIAN"
    mkdir -p "$DEB_DIR/opt/$APP_NAME"
    mkdir -p "$DEB_DIR/usr/share/applications"
    mkdir -p "$DEB_DIR/usr/share/icons/hicolor/128x128/apps"
    
    # Copy application files
    cp -r "$INSTALLER_DIR/staging/opt/$APP_NAME"/* "$DEB_DIR/opt/$APP_NAME/"
    
    # Copy desktop entry and icon
    cp "$RESOURCES_DIR/stereovision3d.desktop" "$DEB_DIR/usr/share/applications/"
    cp "$RESOURCES_DIR/icons/stereovision3d.png" "$DEB_DIR/usr/share/icons/hicolor/128x128/apps/"
    
    # Create control file
    cat > "$DEB_DIR/DEBIAN/control" << EOF
Package: stereovision3d
Version: $APP_VERSION
Section: graphics
Priority: optional
Architecture: amd64
Depends: libopencv-dev (>= 4.5), libpcl-dev (>= 1.10), qt6-base-dev
Maintainer: $APP_VENDOR <$APP_CONTACT>
Description: $APP_DESCRIPTION
 A high-performance C++ application for generating 3D point clouds from
 stereo camera images using GPU acceleration. Features include:
 .
 * Real-time stereo vision processing
 * GPU acceleration with CUDA/HIP
 * Modern Qt6 user interface
 * Professional calibration tools
 * Multiple export formats
Homepage: $APP_URL
EOF

    # Create postinst script
    cat > "$DEB_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database -q
fi

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    gtk-update-icon-cache -q /usr/share/icons/hicolor
fi

# Create symlink in /usr/local/bin
if [ ! -L /usr/local/bin/stereovision3d ]; then
    ln -s /opt/StereoVision3D/bin/stereo_vision_app /usr/local/bin/stereovision3d
fi

echo "StereoVision3D installation completed successfully!"
echo "Run 'stereovision3d' from command line or find it in your applications menu."
EOF

    # Create prerm script
    cat > "$DEB_DIR/DEBIAN/prerm" << 'EOF'
#!/bin/bash
set -e

# Remove symlink
if [ -L /usr/local/bin/stereovision3d ]; then
    rm /usr/local/bin/stereovision3d
fi
EOF

    # Make scripts executable
    chmod 755 "$DEB_DIR/DEBIAN/postinst"
    chmod 755 "$DEB_DIR/DEBIAN/prerm"
    
    # Build package
    dpkg-deb --build "$DEB_DIR" "$PACKAGE_DIR/stereovision3d_${APP_VERSION}_amd64.deb"
    
    print_success "Debian package created: $PACKAGE_DIR/stereovision3d_${APP_VERSION}_amd64.deb"
}

# Create RPM package
create_rpm_package() {
    print_status "Creating RPM package..."
    
    # Create RPM spec file
    cat > "$INSTALLER_DIR/stereovision3d.spec" << EOF
Name:           stereovision3d
Version:        $APP_VERSION
Release:        1%{?dist}
Summary:        $APP_DESCRIPTION
License:        MIT
URL:            $APP_URL
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cmake >= 3.18
BuildRequires:  gcc-c++
BuildRequires:  opencv-devel >= 4.5
BuildRequires:  pcl-devel >= 1.10
BuildRequires:  qt6-qtbase-devel

Requires:       opencv >= 4.5
Requires:       pcl >= 1.10
Requires:       qt6-qtbase

%description
A high-performance C++ application for generating 3D point clouds from
stereo camera images using GPU acceleration. Features include real-time
stereo vision processing, GPU acceleration with CUDA/HIP, modern Qt6
user interface, professional calibration tools, and multiple export formats.

%prep
%setup -q

%build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%{_prefix}
make %{?_smp_mflags}

%install
cd build
%make_install

# Install desktop entry
mkdir -p %{buildroot}%{_datadir}/applications
install -m 644 %{_sourcedir}/../resources/stereovision3d.desktop %{buildroot}%{_datadir}/applications/

# Install icon
mkdir -p %{buildroot}%{_datadir}/icons/hicolor/128x128/apps
install -m 644 %{_sourcedir}/../resources/icons/stereovision3d.png %{buildroot}%{_datadir}/icons/hicolor/128x128/apps/

%files
%{_bindir}/stereo_vision_app
%{_datadir}/applications/stereovision3d.desktop
%{_datadir}/icons/hicolor/128x128/apps/stereovision3d.png

%post
/usr/bin/update-desktop-database &> /dev/null || :
/usr/bin/gtk-update-icon-cache %{_datadir}/icons/hicolor &> /dev/null || :

%postun
/usr/bin/update-desktop-database &> /dev/null || :
/usr/bin/gtk-update-icon-cache %{_datadir}/icons/hicolor &> /dev/null || :

%changelog
* $(date +'%a %b %d %Y') $APP_VENDOR <$APP_CONTACT> - $APP_VERSION-1
- Initial release
EOF

    if command -v rpmbuild &> /dev/null; then
        # Create source tarball
        cd "$PROJECT_ROOT"
        tar -czf "$INSTALLER_DIR/stereovision3d-$APP_VERSION.tar.gz" \
            --exclude='build*' \
            --exclude='packages' \
            --exclude='installer' \
            --exclude='.git*' \
            .
        
        # Build RPM
        mkdir -p "$HOME/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}"
        cp "$INSTALLER_DIR/stereovision3d-$APP_VERSION.tar.gz" "$HOME/rpmbuild/SOURCES/"
        cp "$INSTALLER_DIR/stereovision3d.spec" "$HOME/rpmbuild/SPECS/"
        
        rpmbuild -ba "$HOME/rpmbuild/SPECS/stereovision3d.spec"
        
        # Copy to package directory
        cp "$HOME/rpmbuild/RPMS/x86_64/stereovision3d-$APP_VERSION"*.rpm "$PACKAGE_DIR/"
        
        print_success "RPM package created in $PACKAGE_DIR/"
    else
        print_warning "rpmbuild not available, skipping RPM package creation"
    fi
}

# Create Windows MSI package (requires WiX Toolset)
create_msi_package() {
    print_status "Creating Windows MSI package..."
    
    # Create WiX configuration
    cat > "$INSTALLER_DIR/stereovision3d.wxs" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="$APP_NAME" Language="1033" Version="$APP_VERSION" 
           Manufacturer="$APP_VENDOR" UpgradeCode="12345678-1234-5678-9012-123456789012">
    
    <Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine" />
    
    <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
    
    <MediaTemplate EmbedCab="yes" />
    
    <Feature Id="ProductFeature" Title="$APP_NAME" Level="1">
      <ComponentGroupRef Id="ProductComponents" />
    </Feature>
    
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFilesFolder">
        <Directory Id="INSTALLFOLDER" Name="$APP_NAME" />
      </Directory>
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="$APP_NAME"/>
      </Directory>
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>
    
    <ComponentGroup Id="ProductComponents" Directory="INSTALLFOLDER">
      <Component Id="MainExecutable" Guid="87654321-4321-8765-2109-876543210987">
        <File Id="StereoVisionApp" Source="staging/opt/$APP_NAME/bin/stereo_vision_app.exe" />
      </Component>
      
      <Component Id="StartMenuShortcut" Guid="11111111-2222-3333-4444-555555555555">
        <Shortcut Id="StartMenuShortcut" Directory="ApplicationProgramsFolder"
                  Name="$APP_NAME" Target="[#StereoVisionApp]" WorkingDirectory="INSTALLFOLDER" />
        <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />
        <RegistryValue Root="HKCU" Key="Software\\Microsoft\\$APP_NAME"
                       Name="installed" Type="integer" Value="1" KeyPath="yes" />
      </Component>
      
      <Component Id="DesktopShortcut" Guid="22222222-3333-4444-5555-666666666666">
        <Shortcut Id="DesktopShortcut" Directory="DesktopFolder"
                  Name="$APP_NAME" Target="[#StereoVisionApp]" WorkingDirectory="INSTALLFOLDER" />
        <RegistryValue Root="HKCU" Key="Software\\Microsoft\\$APP_NAME"
                       Name="desktop" Type="integer" Value="1" KeyPath="yes" />
      </Component>
    </ComponentGroup>
  </Product>
</Wix>
EOF

    if command -v candle &> /dev/null && command -v light &> /dev/null; then
        cd "$INSTALLER_DIR"
        candle stereovision3d.wxs
        light stereovision3d.wixobj -out "$PACKAGE_DIR/StereoVision3D-$APP_VERSION.msi"
        print_success "MSI package created: $PACKAGE_DIR/StereoVision3D-$APP_VERSION.msi"
    else
        print_warning "WiX Toolset not available, skipping MSI package creation"
    fi
}

# Create macOS DMG package
create_dmg_package() {
    print_status "Creating macOS DMG package..."
    
    if [[ "$OS" != "macos" ]]; then
        print_warning "Not on macOS, skipping DMG creation"
        return
    fi
    
    APP_BUNDLE="$INSTALLER_DIR/$APP_NAME.app"
    
    # Create app bundle structure
    mkdir -p "$APP_BUNDLE/Contents/MacOS"
    mkdir -p "$APP_BUNDLE/Contents/Resources"
    
    # Copy executable
    cp "$INSTALLER_DIR/staging/opt/$APP_NAME/bin/stereo_vision_app" "$APP_BUNDLE/Contents/MacOS/"
    
    # Create Info.plist
    cat > "$APP_BUNDLE/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>stereo_vision_app</string>
    <key>CFBundleIdentifier</key>
    <string>com.stereovision.stereovision3d</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>$APP_VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$APP_VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF

    # Create DMG
    if command -v hdiutil &> /dev/null; then
        hdiutil create -srcfolder "$APP_BUNDLE" -volname "$APP_NAME" \
                -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDBZ \
                "$PACKAGE_DIR/$APP_NAME-$APP_VERSION.dmg"
        print_success "DMG package created: $PACKAGE_DIR/$APP_NAME-$APP_VERSION.dmg"
    else
        print_warning "hdiutil not available, skipping DMG creation"
    fi
}

# Create AppImage (Linux portable)
create_appimage() {
    print_status "Creating AppImage..."
    
    APPDIR="$INSTALLER_DIR/StereoVision3D.AppDir"
    mkdir -p "$APPDIR"
    
    # Copy application
    cp -r "$INSTALLER_DIR/staging/opt/$APP_NAME"/* "$APPDIR/"
    
    # Create AppRun script
    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/bin/stereo_vision_app" "$@"
EOF
    chmod +x "$APPDIR/AppRun"
    
    # Copy desktop entry and icon
    cp "$RESOURCES_DIR/stereovision3d.desktop" "$APPDIR/"
    cp "$RESOURCES_DIR/icons/stereovision3d.png" "$APPDIR/"
    
    # Download and use appimagetool if available
    if [ ! -f "$INSTALLER_DIR/appimagetool" ]; then
        wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage" \
             -O "$INSTALLER_DIR/appimagetool"
        chmod +x "$INSTALLER_DIR/appimagetool"
    fi
    
    if [ -f "$INSTALLER_DIR/appimagetool" ]; then
        cd "$INSTALLER_DIR"
        ./appimagetool "StereoVision3D.AppDir" "$PACKAGE_DIR/StereoVision3D-$APP_VERSION-x86_64.AppImage"
        print_success "AppImage created: $PACKAGE_DIR/StereoVision3D-$APP_VERSION-x86_64.AppImage"
    else
        print_warning "Could not download appimagetool, skipping AppImage creation"
    fi
}

# Create installation script
create_install_script() {
    cat > "$PACKAGE_DIR/install.sh" << 'EOF'
#!/bin/bash
# Universal installer script for StereoVision3D

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS and package manager
detect_system() {
    if command -v apt-get &> /dev/null; then
        echo "debian"
    elif command -v yum &> /dev/null || command -v dnf &> /dev/null; then
        echo "redhat"
    elif command -v pacman &> /dev/null; then
        echo "arch"
    else
        echo "unknown"
    fi
}

# Install function
install_stereovision3d() {
    local system=$(detect_system)
    
    print_status "Detected system: $system"
    
    case $system in
        debian)
            if [ -f "stereovision3d_*_amd64.deb" ]; then
                print_status "Installing Debian package..."
                sudo dpkg -i stereovision3d_*_amd64.deb || {
                    print_status "Installing dependencies..."
                    sudo apt-get update
                    sudo apt-get install -f -y
                }
                print_success "Installation completed!"
            else
                print_error "Debian package not found"
                exit 1
            fi
            ;;
        redhat)
            if [ -f "stereovision3d-*.rpm" ]; then
                print_status "Installing RPM package..."
                sudo rpm -i stereovision3d-*.rpm || sudo yum install -y stereovision3d-*.rpm
                print_success "Installation completed!"
            else
                print_error "RPM package not found"
                exit 1
            fi
            ;;
        *)
            if [ -f "StereoVision3D-*-x86_64.AppImage" ]; then
                print_status "Using AppImage..."
                chmod +x StereoVision3D-*-x86_64.AppImage
                mkdir -p "$HOME/.local/bin"
                cp StereoVision3D-*-x86_64.AppImage "$HOME/.local/bin/stereovision3d"
                print_success "AppImage installed to ~/.local/bin/stereovision3d"
                print_status "Make sure ~/.local/bin is in your PATH"
            else
                print_error "No compatible package found for your system"
                exit 1
            fi
            ;;
    esac
}

echo "StereoVision3D Installer"
echo "========================"
echo
install_stereovision3d
EOF

    chmod +x "$PACKAGE_DIR/install.sh"
    print_success "Universal install script created"
}

# Main execution
main() {
    print_status "🎯 Professional Package Generation Starting..."
    print_status "Package type: $PACKAGE_TYPE"
    print_status "Target architecture: $TARGET_ARCH"
    print_status "Build type: $BUILD_TYPE"
    echo
    
    setup_directories
    build_application
    create_desktop_entry
    create_application_icon
    
    # Create packages based on type
    case $PACKAGE_TYPE in
        "deb")
            create_deb_package
            ;;
        "rpm")
            create_rpm_package
            ;;
        "msi")
            create_msi_package
            ;;
        "dmg")
            create_dmg_package
            ;;
        "appimage")
            create_appimage
            ;;
        "all")
            if [[ "$OS" == "linux" ]]; then
                create_deb_package
                create_rpm_package
                create_appimage
            elif [[ "$OS" == "macos" ]]; then
                create_dmg_package
            elif [[ "$OS" == "windows" ]]; then
                create_msi_package
            fi
            ;;
        *)
            print_error "Unknown package type: $PACKAGE_TYPE"
            exit 1
            ;;
    esac
    
    create_install_script
    
    echo
    print_success "🎉 Package generation completed!"
    print_status "Packages created in: $PACKAGE_DIR"
    ls -la "$PACKAGE_DIR"
}

# Run main function
main "$@"
