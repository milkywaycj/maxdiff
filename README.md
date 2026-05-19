# maxdiff

Open tools for creating and analyzing MaxDiff studies.

## Which Tool Should I Use?

### Browser Tools

Use the GitHub Pages tools when you want a quick, private, no-install workflow:

- [Analysis tool](https://milkywaycj.github.io/maxdiff/analysis/) - count-based MaxDiff scores with optional bootstrap confidence intervals.
- [Design tool](https://milkywaycj.github.io/maxdiff/design/) - balanced MaxDiff design generation.

Your data stays in your browser. Nothing is uploaded to a server.

The browser analysis tool does **not** run Hierarchical Bayes (HB). HB is too slow and heavy for the Pyodide/WebAssembly browser runtime.

### Windows Desktop Tool

The Windows desktop version also defaults to count-based analysis, which is the recommended analysis type for most MaxDiff studies.

Use the Windows desktop version when you want a local Windows app, or when you specifically need:

- Hierarchical Bayes (HB) estimation.
- Individual-level utilities.
- Larger local analyses without browser runtime limits.
- Follow-up workflows such as latent class segmentation, TURF optimization, or personalization.

Download the latest Windows executable from:

https://github.com/milkywaycj/maxdiff/releases/latest

Windows may show an "unknown publisher" or SmartScreen warning because the executable is not code-signed.

## Hierarchical Bayes Notes

HB support depends on the optional Python libraries in `requirements-hb.txt`: `jax` and `numpyro`, including their normal CPU runtime dependencies.

In the desktop app:

- Count-based analysis is the default and recommended method for most studies.
- If HB dependencies are available, the Hierarchical Bayes option is enabled.
- If HB dependencies are missing, the HB option is disabled.
- HB can take several minutes or longer depending on the number of respondents, items, tasks, iterations, and chains.

For most balanced MaxDiff studies, count-based scores with bootstrap confidence intervals should produce very similar rankings with greater transparency. Use HB when you specifically need individual-level utilities or model-based follow-up analyses.

## Rebuilding the Desktop App

Most users do not need to build anything. Developers can rebuild the Windows executable from source using the instructions in:

[pyinstaller/build_instructions.md](pyinstaller/build_instructions.md)

Install the standard requirements from `requirements.txt`. Install `requirements-hb.txt` as well if you want the rebuilt desktop app to include HB support.

## Repository Layout

- `docs/` - GitHub Pages browser tools.
- `src/` - desktop Python analysis tool.
- `requirements.txt` - standard desktop dependencies.
- `requirements-hb.txt` - optional HB dependencies.
- `pyinstaller/` - Windows executable build notes.
