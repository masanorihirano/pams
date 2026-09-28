Json config
==========================

.. code-block:: json

    {
        "simulation": {
            "markets": ["MarketName", ...],
            "agents": ["AgentName", ...],
            "sessions": [
                {
                    "sessionName": str (or anything that can be converted into str),
                    "iterationSteps": int,
                    "withOrderPlacement": bool,
                    "withOrderExecution": bool,
                    "withPrint": bool,
                    "highFrequencySubmitRate": float (Optional; default 1.0),
                    "maxNormalOrders": int (>=0; Optional; default 1),
                    "maxHighFrequencyOrders": int (>=0; Optional; default 1),
                    "events": ["EventName"] (Optional)
                },
                ...
            ],
            "fundamentalCorrelations": { # Optional
                "pairwise": [
                    ["MarketName1", "MarketName2",  float], # fundamentalVolatility is required in both markets
                    ...
                ]
            },
            "numParallel": int (Optional; default: the number of CPUs - 1; only for MultiThreadAgentParallelRunner and MultiProcessAgentParallelRunner),
        },
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": "MarketName",
            "triggerTime": int,
            "priceChangeRate": float,
            "shockTimeLength": int (Optional, default: 1),
            "enabled": bool (Optional, default: True)
        },
        "PriceLimitRule": {
            "class": "PriceLimitRule",
            "targetMarkets": ["Market"],
            "triggerChangeRate": float required,
            "enabled": bool (Optional, default: True)
        },
        "TradingHaltRule": {
        	"class": "TradingHaltRule",
        	"targetMarkets": ["Market"],
        	"triggerChangeRate": float required,
        	"haltingTimeLength": int required,
        	"enabled": bool (Optional, default: True)
        },
        "OrderMistakeShock": {
            "class": "OrderMistakeShock",
            "target": "Market",
            "triggerTime": int required,
            "priceChangeRate": float required,
            "orderVolume": int required,
            "orderTimeLength": int required,
            "enabled": bool (Optional, default: True)
        },
        "Market": {
            "extends": string (Optional),
            "class": string,
            "tickSize": float,
            "numMarket": int (Optional; default  1),
            "from": int (Optional; cannot be extended),
            "to": int (Optional; cannot be extended; End is included),
            "prefix": str (Optional; default is set to dict key),
            "tickSize": float required,
            "marketPrice": float optional (marketPrice or fundamentalPrice must be specified),
            "fundamentalPrice": float optional (marketPrice or fundamentalPrice must be specified),
            "fundamentalDrift": float (Optional; default: 0.0),
            "fundamentalVolatility": float (Optional; default 0.0),
            "outstandingShares": int optional (default 0)
        },
        "Agents": {
            "class": string,
            "numAgents": int (Optional; default  1),
            "from": int (Optional; cannot be extended),
            "to": int (Optional; cannot be extended; End is included),
            "prefix": str (Optional; default is set to dict key),
            "markets": ["Market", ...] (Required),
            "assetVolume": int (JsonRandom applicable),
            "cashAmount": float (JsonRandom applicable)
        },
        "FCNAgent": {
            "class": "FCNAgent",
            "extends": "Agents",
            "fundamentalWeight": JsonRandomFormat,
            "chartWeight": JsonRandomFormat,
            "noiseWeight": JsonRandomFormat,
            "meanReversionTime":JsonRandomFormat,
            "noiseScale": JsonRandomFormat,
            "timeWindowSize": JsonRandomFormat,
            "orderMargin": JsonRandomFormat,
            "marginType": "fixed" or "normal" (Optional; default fixed)
        },
	    "MarketShareFCNAgents": {
            "class": "MarketShareFCNAgent",
            "extends": "FCNAgent"
        },
        "ArbitrageAgent": {
            "class": "ArbitrageAgent",
            "extends": "Agents",
            "orderVolume": int,
            "orderThresholdPrice": float,
            "orderTimeLength": int (Optional, default 1),
        },
        "MarketMakerAgent": {
            "class": "MarketMakerAgent",
            "extends": "Agents",
            "targetMarket": string required,
            "netInterestSpread": float required,
            "orderTimeLength": int optional; default 2,
        }
    }

Naming of markets and agents
----------------------------

Each market / agent gets a name built from ``prefix`` (default: the dict key) as follows:

- Neither ``numMarkets`` / ``numAgents`` nor ``from`` / ``to`` specified, or the count is 1: the name is ``prefix`` itself (e.g. ``Market``).
- ``numMarkets`` / ``numAgents`` greater than 1: the names are ``prefix`` + index, starting from 0. If ``prefix`` is not specified, ``-`` is inserted (e.g. ``Market-0``, ``Market-1``, ...).
- ``from`` and ``to``: both are required, ``to`` is inclusive and must be greater than or equal to ``from``. The names are always ``prefix`` + index for every index from ``from`` to ``to``, even when ``from`` equals ``to``. If ``prefix`` is not specified and the range has more than one element, ``-`` is inserted (e.g. ``from: 0, to: 1`` gives ``Market-0``, ``Market-1``; ``from: 5, to: 5`` gives ``Market5``).
