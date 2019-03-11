# Contribution guide for machina

This file contains guidance for contributing to machina.

We welcome any contibutions.
 + Star this repository.
 + Open issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
 + Send pull requests about fixing bug, adding new features, adding new algorithms etc.


# Coding style

We use PEP8. Test script checks whether all scripts are passing a PEP8 test or not. You should apply `autopep8 -i edited_script` before committing.

When we name variable which means number of hoge, we use `num_hoge`.

## Abbreviations
We use many abbreviations, because many words of the technical jargon in Reinforcement Learning are just too long.
Please use these abbreviations below. You can add 's' to end of the word for the plural form.

```
episode -> epi
action -> ac
observation -> ob
reward -> rew
policy -> pol
value function -> vf
q function -> qf
dynamics model -> dm
learning rate -> lr
lambda -> lam
policy gradient -> pg
action gradient -> ag
soft actor critic -> sac
entropy -> ent
probabilistic distribution -> pd
log likelihood -> llh
target -> targ
Kullback-Leibler -> kl
```

# Test
We use Tracis-CI for testing code in machina. Please check if the status of the tests is shown as passing.
