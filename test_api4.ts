import fetch from 'node-fetch';

(async () => {
    try {
        const key = process.env.RAPIDAPI_KEY;
        const alexRes = await fetch("https://aliexpress-datahub.p.rapidapi.com/item_search_v2?q=watch", {
            headers: {
                'x-rapidapi-key': key!,
                'x-rapidapi-host': 'aliexpress-datahub.p.rapidapi.com'
            }
        });
        console.log("Status:", alexRes.status);
        console.log("JSON:", await alexRes.text());
    } catch (e) {
        console.error(e);
    }
})();
