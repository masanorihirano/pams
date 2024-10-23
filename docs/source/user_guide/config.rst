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
            "numParallel": int (Optional; default 1; only for MultiThreadedRunner),
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
