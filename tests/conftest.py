"""Shared pytest fixtures."""
import jax
import jax.numpy as jnp
import pytest

from surfacehop_jax import TullyModel1, TullyModel2, TullyModel3


@pytest.fixture
def tully1():
    return TullyModel1()


@pytest.fixture
def tully2():
    return TullyModel2()


@pytest.fixture
def tully3():
    return TullyModel3()


@pytest.fixture
def key():
    return jax.random.PRNGKey(20260515)
