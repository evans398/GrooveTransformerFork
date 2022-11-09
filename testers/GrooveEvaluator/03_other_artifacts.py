if __name__ == "__main__":

    # Load Evaluator using full path with extension
    from eval.GrooveEvaluator import load_evaluator
    evaluator_test_set = load_evaluator("testers/GrooveEvaluator/examples/test_set_full_colorful_sweep_41.Eval.bz2")

    # get piano rolls
    piano_rolls = evaluator_test_set.get_piano_rolls(
        prepare_for_wandb=True,
        save_path="testers/GrooveEvaluator/misc/GrooveEvaluator/piano_rolls.html")

    # get audios - separate files for ground truth and predictions
    audio_tuples = evaluator_test_set.get_audio_tuples(
        prepare_for_wandb=True,
        sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
        save_directory="testers/GrooveEvaluator/misc/GrooveEvaluator/audios",
        concatenate_gt_and_pred=False)

    # get audios - a single file containing ground truth and predictions with a 1sec silence in between
    audio_tuples = evaluator_test_set.get_audio_tuples(
        prepare_for_wandb=True,
        sf_path="hvo_sequence/soundfonts/Standard_Drum_Kit.sf2",
        save_directory="testers/GrooveEvaluator/misc/GrooveEvaluator/audios",
        concatenate_gt_and_pred=True)

    # store midi files
    evaluator_test_set.export_to_midi(need_gt=True, need_pred=True, directory="testers/GrooveEvaluator/misc/GrooveEvaluator/midi")