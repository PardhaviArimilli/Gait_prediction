import argparse
import os
from src.serve.ensemble_infer import ensemble_and_postprocess
from src.serve.summarize import main as summarize_main
import sys


def run():
	p = argparse.ArgumentParser(description="Run ensemble inference and summaries")
	p.add_argument("--defog", default="test/defog")
	p.add_argument("--tdcs", default="test/tdcsfog")
	p.add_argument("--out", default="artifacts/infer")
	p.add_argument("--ckpt_cnn", default="artifacts/checkpoints/cnn_bilstm_fold*_best.pt")
	p.add_argument("--ckpt_tcn", default="artifacts/checkpoints/tcn_fold*_best.pt")
	p.add_argument("--export_web", action="store_true", help="Copy summaries to website/netlify for dev")
	args = p.parse_args()

	os.makedirs(args.out, exist_ok=True)

	# defog
	out_defog = os.path.join(args.out, "defog", "ens")
	ensemble_and_postprocess(args.defog, out_defog, args.ckpt_cnn, args.ckpt_tcn)
	# summary defog
	sys.argv = ["", "--dir", out_defog, "--out", os.path.join(out_defog, "summary.json"), "--csv", os.path.join(out_defog, "summary.csv")]
	summarize_main()

	# tdcs
	out_tdcs = os.path.join(args.out, "tdcs", "ens")
	ensemble_and_postprocess(args.tdcs, out_tdcs, args.ckpt_cnn, args.ckpt_tcn)
	# summary tdcs
	sys.argv = ["", "--dir", out_tdcs, "--out", os.path.join(out_tdcs, "summary.json"), "--csv", os.path.join(out_tdcs, "summary.csv")]
	summarize_main()

	# optional export to website
	if args.export_web:
		import shutil
		webdir = os.path.join("website", "netlify")
		os.makedirs(webdir, exist_ok=True)
		shutil.copyfile(os.path.join(out_defog, "summary.json"), os.path.join(webdir, "summary_defog.json"))
		shutil.copyfile(os.path.join(out_defog, "summary.csv"), os.path.join(webdir, "summary_defog.csv"))
		shutil.copyfile(os.path.join(out_tdcs, "summary.json"), os.path.join(webdir, "summary_tdcs.json"))
		shutil.copyfile(os.path.join(out_tdcs, "summary.csv"), os.path.join(webdir, "summary_tdcs.csv"))


if __name__ == "__main__":
	run()
