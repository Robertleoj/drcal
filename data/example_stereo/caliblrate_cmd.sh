drcal-calibrate-cameras \
    --focal 800 \
    --lensmodel LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=25_Ny=17_fov_x_deg=80 \
    --corners-cache ./data/example_stereo/corners.vnl \
    --object-spacing 0.04 \
    --object-width-n 13 \
    --object-height-n 8 \
    --imagersize 1280 800 \
    --outdir data/example_stereo \
    'data/example_stereo/*_000.png' 'data/example_stereo/*_001.png'