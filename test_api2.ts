import fetch from "node-fetch";

(async () => {
    try {
        const key = process.env.RAPIDAPI_KEY;
        const alibabaRes = await fetch("https://alibaba-datahub.p.rapidapi.com/item_search?q=watch&page=1", {
            headers: {
                'x-rapidapi-key': key!,
                'x-rapidapi-host': 'alibaba-datahub.p.rapidapi.com'
            }
        });
        const aliJson = await alibabaRes.json();
        console.log("Alibaba item 0:", JSON.stringify(aliJson.result?.resultList?.[0], null, 2));

        const alexRes = await fetch("https://aliexpress-datahub.p.rapidapi.com/item_search?q=smartwatch&page=1", {
            headers: {
                'x-rapidapi-key': key!,
                'x-rapidapi-host': 'aliexpress-datahub.p.rapidapi.com'
            }
        });
        const alexJson = await alexRes.json();
        console.log("AliExpress whole:", JSON.stringify(alexJson).substring(0, 500));
        console.log("AliExpress item 0:", JSON.stringify(alexJson.result?.resultList?.[0], null, 2));
    } catch (e) {
        console.error(e);
    }
})();
