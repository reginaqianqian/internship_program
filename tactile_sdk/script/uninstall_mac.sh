#!/bin/bash

PKG_ID="com.sharpa.tactile_sdk.Unspecified"

# Remove all files installed by the package
pkgutil --only-files --files "$PKG_ID" | while read -r file; do
  sudo rm -fv "/$file"
done

# Remove empty directories
pkgutil --only-dirs --files "$PKG_ID" | while read -r dir; do
  sudo rmdir -pv "/$dir" 2>/dev/null || true
done

# Forget package registration
sudo pkgutil --forget "$PKG_ID"
