from pyformlang.cfg import CFG, Production, Terminal, Variable

from project.tensor_cfpq import cfg_to_rsm


def test_cfg_to_rsm():
    # S -> "|"
    s = Variable("S")
    pipe = Terminal("|")
    cfg = CFG(
        variables={s},
        terminals={pipe},
        start_symbol=s,
        productions={Production(s, [pipe])},
    )

    cfg_to_rsm(cfg)
