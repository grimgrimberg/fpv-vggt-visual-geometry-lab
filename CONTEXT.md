# Context

This project analyzes already-published FPV footage as offline historical media.
Its domain language must keep visual-geometry review separate from operational
drone analysis.

## Glossary

### Dataset catalog

The remote repository metadata that lists dataset video identity, dates,
descriptions, thumbnails, and download links. The catalog is not a license grant
for third-party media.

### Source description

The original description text supplied by the dataset catalog. It is provenance
metadata only and must not become an analytic target label, a map feature, or an
operational category.

### Local media cache

Downloaded media files held under ignored local paths for analysis. Local media
cache contents are not redistributed by this repository.

### Flight segment

The portion of a clip selected for relative visual-geometry analysis. V1 may
propose draft segments automatically, but downstream analysis requires a human
accepted segment.

### VGGT prediction bundle

The repo-owned imported artifact format for VGGT outputs. It stores relative
camera data, point-cloud data, frame alignment, and provenance without requiring
VGGT to be installed.

### Relative camera path

The camera-center trajectory reconstructed in the VGGT coordinate frame. It is
scale ambiguous and rotation ambiguous by default. It is not a real-world path
in meters.

### Reconstruction reliability

A transparent diagnostic estimate of whether the imported relative
reconstruction is usable for review or conservative descriptors. It is not a
truth claim about the event.

### Scale-free path descriptor

A relative path-shape metric that avoids meters, geolocation, target-relative
features, and physical speed claims.

### TTVE

Time-to-visible-terminal-event. In this project it is future-only retrospective
benchmarking on historical edited clips, not live prediction or operational
warning.

### Relative pose smoother

An optional reliability-gated diagnostic that smooths imported VGGT-frame poses.
It is not drone dynamics, guidance, control, or future trajectory prediction.
