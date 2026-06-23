# Models

Model files are not tracked in the source repository.

The default v2.0.0 model is shipped in the GitHub Release package:

```text
models/halfkp_wp_h512_e15_500m_clip30_quant.txt
```

Place that file in this directory when running a source build with the default
`neural_halfkp_quant_accum` backend.

Post-release local testing also produced a stronger development candidate:

```text
models/halfkp_wp_h512_e6_all_data_clip30_quant.txt
```

That model is not part of the v2.0.0 source tree or release package unless a
future release explicitly ships it.
