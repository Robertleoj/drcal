r"""Converts a .cahvor-formatted camera model to the .cameramodel file format

SYNOPSIS

  $ drcal-from-cahvor model1.cahvor model2.cahvor
  Wrote model1.cameramodel
  Wrote model2.cameramodel

This tool converts the given model(s) to the cameramodel file format. No changes
to the content are made; this is purely a format converter (the
drcal-convert-lensmodel tool fits different lens models instead). Model
filenames are given on the commandline. Output is written to the same directory,
with the same filename, but with a .cameramodel extension.

If the model is omitted or given as "-", the input is read from standard input,
and the output is written to standard output

"""

import sys
import argparse
import os
import drcal


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="""By default existing files are not overwritten. Pass --force to overwrite them
                        without complaint""",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        type=lambda d: d
        if os.path.isdir(d)
        else parser.error(
            "--outdir requires an existing directory as the arg, but got '{}'".format(d)
        ),
        help="""Directory to write the output models into. If omitted, we write the output
                        models to the same directory as the input models""",
    )
    parser.add_argument(
        "model", default=["-"], nargs="*", type=str, help="""Input camera model"""
    )

    return parser.parse_args()


def main():
    args = parse_args()

    Nstdin = sum(1 for m in args.model if m == "-")
    if Nstdin > 1:
        print(
            f"At most one model can be read from standard input ('-'), but I got {Nstdin}",
            file=sys.stderr,
        )
        sys.exit(1)

    for model in args.model:
        if model == "-":
            try:
                m = drcal.cameramodel(model)
            except KeyboardInterrupt:
                sys.exit(1)
            except Exception as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            m.write(sys.stdout)
        else:
            base, extension = os.path.splitext(model)
            if extension.lower() == ".cameramodel":
                print(
                    "Input file is already in the cameramodel format (judging from the filename). Doing nothing",
                    file=sys.stderr,
                )
                sys.exit(0)

            if args.outdir is not None:
                base = args.outdir + "/" + os.path.split(base)[1]
            filename_out = base + ".cameramodel"
            if not args.force and os.path.exists(filename_out):
                print(
                    f"Target model '{filename_out}' already exists. Doing nothing with this model. Pass -f to overwrite",
                    file=sys.stderr,
                )
            else:
                m = drcal.cameramodel(model)
                m.write(filename_out)
                print("Wrote " + filename_out)
