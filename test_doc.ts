import fetch from 'node-fetch';
(async () => {
    try {
        const res = await fetch("https://rapidapi.com/ecommdatahub/api/aliexpress-datahub/playground/apiendpoint_6aa9c114-c780-433a-8033-46ccd61a156c");
        const html = await res.text();
        
        // Find text near "endpoint" or "query"
        const snippetInfo = html.match(/.{0,200}item_search.{0,200}/gi) || [];
        console.log("HTML snippets:");
        snippetInfo.forEach(s => console.log(s));
    } catch(e) {
        console.error(e);
    }
})();
