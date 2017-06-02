for scenario in bptt tbptt msbptt mstbptt; do
ipython3 --pdb -- graph.py --scenario $scenario --interactive false --verbose-debug
done
