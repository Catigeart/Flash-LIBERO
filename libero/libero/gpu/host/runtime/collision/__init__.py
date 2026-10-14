"""P06-C collision runtime: complete ordered collision device chain.

Ownership namespace `collision runtime`; written only under the P06-C
execution exclusive prefixes. Stages: bounds -> SAP -> driver body-pair
dedup/exclude -> BVH geom invocation -> dispatch -> detector registry ->
finalize, with device counts driving capacity and first-error routing.
"""
