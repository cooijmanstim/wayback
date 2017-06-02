for scenario in bptt tbptt msbptt msbptt; do
ipython --pdb -- graph.py --scenario $scenario --interactive false --verbose-debug
done
