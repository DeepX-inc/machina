# Contribution guide for machina

This is a guidance for contributing to machina.

We are welcome for any contibutions.
 + Star this repository.
 + Open issues about questions, bugs, installation problems, feature requests, algorithm requests etc.
 + Send pull requests about fixing bug, adding new features, adding new algorithms etc.


# Coding style

We use PEP8. Test script checks whether all scripts are passing PEP8 or not. You should apply `autopep8 -i edited_script` before committing.

When we name variable which means number of hoge, we use `num_hoge`.

## Abbreviation
We use many abbreviation, because many of technical jargons in Reinforcement Learning are too long.
Please use these abbreviation below. You can add 's' to end of the word for plural form.

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
```

# Test
We use Tracis-CI for test. Please check test status to be passed.
