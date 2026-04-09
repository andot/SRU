#!/usr/bin/env python3
"""Shortcut: call the unified export tool for NCNN (pnnx)."""
import sys

from export import main as export_main


def main():
    sys.argv = [sys.argv[0], 'ncnn'] + sys.argv[1:]
    export_main()


if __name__ == '__main__':
    main()
