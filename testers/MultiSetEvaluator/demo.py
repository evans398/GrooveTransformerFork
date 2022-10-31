from eval.GrooveEvaluator import load_evaluator
from eval.MultiSetEvaluator import MultiSetEvaluator

# prepare input data
eval_1 = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_robust_sweep_29.Eval.bz2")
eval_2 = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

# ignore_feature_keys = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
ignore_feature_keys = None

# construct MultiSetEvaluator
msEvaluator = MultiSetEvaluator(
    groove_evaluator_sets={"Model 1": eval_1, "Model 2": eval_2, "Model 3": eval_2},
    # { "groovae": eval_1, "Model 1": eval_2, "Model 2": eval_3 },  # { "groovae": eval_1}
    ignore_feature_keys=None,  # ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]
    reference_set_label="GT",
    anchor_set_label=None  # "groovae"
)

# dump MultiSetEvaluator
msEvaluator.dump("testers/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

# load MultiSetEvaluator
from eval.MultiSetEvaluator import load_multi_set_evaluator
msEvaluator = load_multi_set_evaluator("testers/MultiSetEvaluator/misc/inter_intra_evaluator.MSEval.bz2")

# save statistics
msEvaluator.save_statistics_of_inter_intra_distances(dir_path="testers/MultiSetEvaluator/misc/multi_set_evaluator")

# save inter intra pdf plots
iid_pdfs_bokeh = msEvaluator.get_inter_intra_pdf_plots(
    filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/iid_pdfs.html")

# save kl oa plots
KL_OA_plot = msEvaluator.get_kl_oa_plots(filename="testers/MultiSetEvaluator/misc/multi_set_evaluator")

# get pos neg hit score plots
pos_neg_hit_score_plots = msEvaluator.get_pos_neg_hit_score_plots(
    filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/pos_neg_hit_scores.html")

# get velocity distribution plots
velocity_distribution_plots = msEvaluator.get_velocity_distribution_plots(
    filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/velocity_distributions.html")

# get offset distribution plots
offset_distribution_plots = msEvaluator.get_offset_distribution_plots(
    filename="testers/MultiSetEvaluator/misc/multi_set_evaluator/offset_distributions.html")

