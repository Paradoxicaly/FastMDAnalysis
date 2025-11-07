# FastMDAnalysis/src/fastmdanalysis/utils/options.py
"""
Options Forwarding Utility

Provides permissive options passthrough with:
- Introspection of target signatures to forward accepted kwargs
- Alias mapping (e.g., ref → reference, atoms → atom_indices)
- Pre-hooks (e.g., align: true → traj.superpose(...))
- Post-hooks (e.g., per_residue: true → residue aggregation)
- Strictness mode (log unknown keys or raise errors)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import inspect
import logging
import warnings

logger = logging.getLogger(__name__)

__all__ = [
    "OptionsForwarder",
    "forward_options",
    "apply_alias_mapping",
]


class OptionsForwarder:
    """
    Handles permissive options forwarding with alias mapping and hooks.
    
    Parameters
    ----------
    aliases : dict, optional
        Mapping of alias names to canonical parameter names.
        Example: {"ref": "reference_frame", "atoms": "atom_indices"}
    pre_hooks : dict, optional
        Mapping of option names to pre-processing functions.
        Functions should accept (value, context) and return processed value.
    post_hooks : dict, optional
        Mapping of option names to post-processing functions.
        Functions should accept (result, context) and return modified result.
    strict : bool
        If True, raise errors for unknown options. If False, log warnings.
    """
    
    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        pre_hooks: Optional[Dict[str, Callable]] = None,
        post_hooks: Optional[Dict[str, Callable]] = None,
        strict: bool = False,
    ):
        self.aliases = aliases or {}
        self.pre_hooks = pre_hooks or {}
        self.post_hooks = post_hooks or {}
        self.strict = strict
        
    def apply_aliases(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply alias mappings to options dictionary.
        
        Parameters
        ----------
        options : dict
            Raw options from user.
            
        Returns
        -------
        dict
            Options with aliases resolved to canonical names.
        """
        resolved = {}
        for key, value in options.items():
            canonical = self.aliases.get(key, key)
            if canonical in resolved and canonical != key:
                # Both alias and canonical provided
                msg = f"Both alias '{key}' and canonical '{canonical}' provided; using canonical"
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
                    continue
            resolved[canonical] = value
        return resolved
    
    def forward_to_callable(
        self,
        callable_obj: Callable,
        options: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Forward options to a callable, keeping only accepted parameters.
        
        Parameters
        ----------
        callable_obj : callable
            Target function or class constructor.
        options : dict
            Options to forward (already with aliases resolved).
            
        Returns
        -------
        forwarded : dict
            Options accepted by the callable.
        dropped : list
            Names of options that were dropped.
        """
        try:
            sig = inspect.signature(callable_obj)
        except (ValueError, TypeError):
            # Can't inspect signature, forward all
            logger.debug("Cannot inspect signature of %s, forwarding all options", callable_obj)
            return options, []
        
        # Get accepted parameter names
        accepted = set()
        has_var_keyword = False
        
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
            elif param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted.add(name)
        
        # If callable accepts **kwargs, forward all
        if has_var_keyword:
            return options, []
        
        # Otherwise, filter to accepted parameters
        forwarded = {k: v for k, v in options.items() if k in accepted}
        dropped = [k for k in options.keys() if k not in accepted]
        
        return forwarded, dropped
    
    def process_options(
        self,
        options: Dict[str, Any],
        callable_obj: Optional[Callable] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Full pipeline: apply aliases, run pre-hooks, forward to callable.
        
        Parameters
        ----------
        options : dict
            Raw options from user.
        callable_obj : callable, optional
            Target callable for signature introspection.
        context : dict, optional
            Additional context for hooks (e.g., trajectory object).
            
        Returns
        -------
        forwarded : dict
            Processed options ready to pass to callable.
        hook_data : dict
            Data from pre-hooks that may be needed later.
        """
        context = context or {}
        hook_data = {}
        
        # Step 1: Apply aliases
        resolved = self.apply_aliases(options)
        
        # Step 2: Run pre-hooks
        processed = {}
        for key, value in resolved.items():
            if key in self.pre_hooks:
                try:
                    result = self.pre_hooks[key](value, context)
                    if isinstance(result, tuple):
                        # Hook can return (processed_value, extra_data)
                        processed[key], hook_data[key] = result
                    else:
                        processed[key] = result
                except Exception as e:
                    msg = f"Pre-hook for '{key}' failed: {e}"
                    if self.strict:
                        raise RuntimeError(msg) from e
                    else:
                        logger.warning(msg)
                        processed[key] = value
            else:
                processed[key] = value
        
        # Step 3: Forward to callable if provided
        if callable_obj is not None:
            forwarded, dropped = self.forward_to_callable(callable_obj, processed)
            
            if dropped:
                msg = f"Unknown options (not accepted by callable): {dropped}"
                if self.strict:
                    raise ValueError(msg)
                else:
                    logger.info(msg)
        else:
            forwarded = processed
            
        return forwarded, hook_data


def apply_alias_mapping(options: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    """
    Simple utility to apply alias mappings to options.
    
    Parameters
    ----------
    options : dict
        Raw options from user.
    aliases : dict
        Mapping of alias names to canonical names.
        
    Returns
    -------
    dict
        Options with aliases resolved.
    """
    resolved = {}
    for key, value in options.items():
        canonical = aliases.get(key, key)
        resolved[canonical] = value
    return resolved


def forward_options(
    target: Callable,
    options: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Simple utility to forward options with alias mapping.
    
    Parameters
    ----------
    target : callable
        Target function or class constructor.
    options : dict
        Options to forward.
    aliases : dict, optional
        Alias mappings.
    strict : bool
        If True, raise errors for unknown options.
        
    Returns
    -------
    dict
        Filtered options ready to pass to target.
    """
    forwarder = OptionsForwarder(aliases=aliases, strict=strict)
    forwarded, _ = forwarder.process_options(options, callable_obj=target)
    return forwarded
