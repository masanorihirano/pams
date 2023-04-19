Json config
==========================

.. code-block:: json

    {
        "simulation": {
            "markets": ["SpotMarket-N"],
            "agents": ["FCNAgents"],
            "sessions": [
                {	"sessionName": 0,
                    "iterationSteps": 100,
                    "withOrderPlacement": true,
                    "withOrderExecution": false,
                    "withPrint": true,
                    "highFrequencySubmitRate": float
                    "maxNormalOrders": int (>=0),
                    "maxHighFrequencyOrders": int (>=0)
                },
                {	"sessionName": 1,
                    "iterationSteps": 500,
                    "withOrderPlacement": true,
                    "withOrderExecution": true,
                    "withPrint": true,
                    "highFrequencySubmitRate": float
                    "maxNormalOrders": int (>=0),
                    "maxHighFrequencyOrders": int (>=0)
                    "events": ["FundamentalPriceShock"]
                }
            ],
            "fundamentalCorrelations": {
                "pairwise": [
                    ["SpotMarket-0", "SpotMarket-1",  0.9],
                    ["SpotMarket-0", "SpotMarket-2", -0.1]
                ]
            }
        },
        "FundamentalPriceShock": {
            "class": "FundamentalPriceShock",
            "target": "SpotMarket-1",
            "triggerTime": 0,
            "priceChangeRate": -0.1,
            "shockTimeLength": int (Optional, default: 1),
            "enabled": bool (Optional, default: True)
        },
        "Market": {
            "class": "string required",
            "tickSize": float required,
            "numMarket": int (Optional; default  1),
            "from": int (Optional; cannot be extended),
            "to": int (Optional; cannot be extended; End is included),
            "prefix": str (Optional; default is set to dict key),
            "tickSize": float required,
            "marketPrice": float optional (marketPrice or fundamentalPrice must be specified),
            "fundamentalPrice": float optional (marketPrice or fundamentalPrice must be specified),
            "fundamentalDrift": float (Optional; default: 0.0),
            "fundamentalVolatility": float (Optional; default 0.0),
            "outstandingShares": int optional (default 0),
        },
        "MarketBase": {
            "class": "Market",
            "tickSize": 0.00001,
            "marketPrice": 300.0
        },
        "SpotMarket-N": {
            "prefix": "SpotMarket-",
            "numMarkets": 3,
            "extends": "MarketBase"
        },
        "Agents": {
            "class": "string required",
            "numAgents": int (Optional; default  1),
            "from": int (Optional; cannot be extended),
            "to": int (Optional; cannot be extended; End is included),
            "prefix": str (Optional; default is set to dict key),
            "markets": ["Market"] (Required),
            "assetVolume": int (JsonRandom applicable),
            "cashAmount": float (JsonRandom applicable)
        },
        "FCNAgent": {
            "class": "FCNAgent",
            "extends": "Agents",
            "fundamentalWeight": {"expon": [1.0]},
            "chartWeight": {"expon": [0.0]},
            "noiseWeight": {"expon": [1.0]},
            "meanReversionTime":{"uniform":[50,100]},
            "noiseScale": 0.001,
            "timeWindowSize": [100, 200],
            "orderMargin": [0.0, 0.1],
            "marginType": "fixed" or "normal" (Optional)
        },
        "ArbitrageAgent": {
            "class": "ArbitrageAgent",
            "extends": "Agents",
            "orderVolume": int,
            "orderThresholdPrice": float,
            "orderTimeLength": int (Optional, default 1),
        },
    }
