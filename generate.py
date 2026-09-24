"""
Eclipse from the command line: continue one prompt and print the result.

    venv/bin/python generate.py "Once upon a time, there was a tiny robot who wanted to"
    venv/bin/python generate.py "Lily found a shiny key." --temperature 0      # greedy
    venv/bin/python generate.py "Tom was sad." --top-k 40 --seed 1337 --max-new-tokens 200
    venv/bin/python generate.py --info

The prompt and its continuation go to stdout; details (seed, token counts,
why generation stopped) go to stderr.
"""

import argparse
import json
import sys

from eclipse.inference import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K,
                               Eclipse, EclipseError, GenerationSettings)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Continue a prompt with Eclipse")
    parser.add_argument("prompt", nargs="?", help="the beginning of a story")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="0 = greedy (always the most likely token)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="sample among the k most likely tokens; 0 = no top-k")
    parser.add_argument("--seed", type=int, default=None, help="reproduce a sampled output")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path (default: $ECLIPSE_CHECKPOINT, else "
                             "checkpoints/sr_data200mb_s1337_best.pt)")
    parser.add_argument("--device", default="cpu", help="cpu (default), mps, or auto")
    parser.add_argument("--skip-provenance-check", action="store_true",
                        help="use weights even if they cannot be matched to the research record")
    parser.add_argument("--info", action="store_true",
                        help="print what was loaded as JSON and exit")
    args = parser.parse_args(argv)
    if args.prompt is None and not args.info:
        parser.error("a prompt is required (or --info)")

    try:
        engine = Eclipse.load(args.checkpoint, device=args.device,
                              verify_provenance=not args.skip_provenance_check)
        if args.info:
            print(json.dumps(engine.info.to_dict(), indent=1))
            return 0
        settings = GenerationSettings(max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature,
                                      top_k=args.top_k or None, seed=args.seed)
        result = engine.generate(args.prompt, settings)
    except EclipseError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(result.prompt + result.completion)
    stop = ("end-of-story token" if result.stop_reason == "eos"
            else f"{settings.max_new_tokens}-token limit")
    print(f"[{result.generated_tokens} tokens, stopped at the {stop}, seed {result.seed}, "
          f"{result.elapsed_seconds:.2f}s]", file=sys.stderr)
    if result.prompt_unknown_tokens:
        print(f"[note: {result.prompt_unknown_tokens} prompt character(s) were never seen in "
              "training and were encoded as <UNK>]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
