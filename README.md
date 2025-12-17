# B.Tech Minor Project

**Ideas:**

Synthetic Tasks Generation:

LLM (ORACLE MODEL) will generate task, task solution, task verification scripts (we will need an sandbox environment which can be reset quickly).

Issues:

Accuracy: We can never be completely sure (use as large a model as you can with chain of thought).

Bias: How we ensure we have the right variety of tasks. We want to start easy and go to tough slowly (very slowly).
      
      First command wise balance (weighted)
      Second task wise balance (weighted) (Here task implies file system, network, firewall, systemd, logs, setup, etc).

Once we have a large synthetic dataset to train our model on.

we can take the ready made dataset for now for the ppt (just have the workflow and script ready to show for something).


RL Training: (Both PPO and GRPO will be tried and reported)

have the model warmed up with supervised fine tuning.

have it solve the tasks multiple time.

have a complex reward function.

set of solutions -> set of rewards.

gradient computation / policy application.

update to the weights (LoRa) (try full one on 500M).

continuously evaluate on test set and train set.

keep checking for overfiting and underfitting.

Some kind of droput if too much overfiting (sparse data).

Increase the dataset veriety and depth to avoid the above.

## Links
 - https://huggingface.co/westenfelder/
 - https://github.com/westenfelder/NL2SH
 - https://github.com/TellinaTool/nl2bash
 - https://github.com/IBM/nl2bash-eabench
 - https://github.com/laude-institute/terminal-bench
 - https://github.com/nokia/nlc2cmd-submission-hubris
 - https://github.com/magnumresearchgroup/Magnum-NLC2CMD
 - https://www.kaggle.com/code/ksmooi/fine-tuning-qwen2-5-3b-instruct-grpo-peft
 - https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
 - https://gist.github.com/qunash/820c86d1d267ec8051d9f68b4f4bb656

 - https://huggingface.co/datasets/deathbyknowledge/shell-tasks
 - https://github.com/deathbyknowledge/sos


# Papers
 - https://arxiv.org/pdf/2507.23751 (FOR RL TASK SYN/GEN)
