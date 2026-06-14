import fetch from 'node-fetch';

(async () => {
    try {
        const key = process.env.RAPIDAPI_KEY;
        const endpoints = ["item_search", "item_search_2", "item_search_3", "item_search_4", "item_search_5"];
        for (const ep of endpoints) {
            const alexRes = await fetch(`https://aliexpress-datahub.p.rapidapi.com/${ep}?q=watch&page=1`, {
                headers: {
                    'x-rapidapi-key': key!,
                    'x-rapidapi-host': 'aliexpress-datahub.p.rapidapi.com'
                }
            });
            const text = await alexRes.text();
            console.log(`Endpoint ${ep} length ${text.length}: `, text.substring(0, 100));
        }
    } catch (e) {
        console.error(e);
    }
})();
