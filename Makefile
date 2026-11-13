.PHONY: install test demo smoke audit ablation sensitivity
install:
	python -m pip install -e ".[rl,test]"
test:
	python -m pytest -q
demo:
	python -m gameedgealloc.cli demo --users 10 --slots 15 --output results/demo
smoke:
	python -m gameedgealloc.cli experiment --config configs/smoke.yaml --output results/smoke
audit:
	python -m gameedgealloc.verification --users 20 --output results/equilibrium_numerical_check.json
ablation:
	python -m gameedgealloc.ablations --mode ablation --config configs/ablation.yaml --output results/ablations
sensitivity:
	python -m gameedgealloc.ablations --mode sensitivity --config configs/ablation.yaml --output results/sensitivity
