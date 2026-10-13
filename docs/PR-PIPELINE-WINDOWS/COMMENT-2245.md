# Comment draft for #2245 (optional second comment - draft only)

Target: https://github.com/Niko1221/Strata/issues/2245 (open).  Posting:

```sh
gh issue comment 2245 --repo Niko1221/Strata --body-file work/scion/docs/PR-PIPELINE-WINDOWS/COMMENT-2245.md
```

---8<--- text starts here ---8<---

One more gfx1100 data point for the "not universal" side: 2x RX 7900 XT,
**Linux, ROCm 7.2**, engine 0.1.44 (plus a local branch that only adds artifact
support - no pipeline changes), `--pipeline-windows 2 --spec 4`, drafter on the
last card (gfx11 keeps that placement): ~30 requests ran clean - 4K contexts,
128-token outputs on the bench protocol, one 400-token generation - and the
`STRATA_MTP_STAGE0=1` arm (drafter on the first card) ran clean too.

At your ~0.7 per 100 rate that is a data point, not a claim that the stall is
absent.  The useful part may be the conditions: Linux/ROCm 7.2 rather than
Windows/0.1.43.1, short contexts, and the default `--short-read 64` in use while
the reported failure is at the short-read tail.
