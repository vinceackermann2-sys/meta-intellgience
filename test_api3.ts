import fetch from 'node-fetch';

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
        
        function extractItemsList(data: any): any[] {
            let largestArray: any[] = [];
            function searchArray(obj: any) {
              if (Array.isArray(obj)) {
                if (obj.length > largestArray.length && typeof obj[0] === 'object') {
                  largestArray = obj;
                }
              } else if (obj !== null && typeof obj === 'object') {
                for (const key in obj) {
                  searchArray(obj[key]);
                }
              }
            }
            searchArray(data);
            return largestArray;
        }

        const items = extractItemsList(aliJson);
        const parsed = items.slice(0, 5).map((i: any) => {
          const itemData = i.item ? i.item : i;
          let minOrder = itemData.min_num || itemData.minOrder || itemData.moq || itemData.minPurchaseNum || "1";
          if (itemData.sku?.def?.quantityModule?.minOrder?.quantityFormatted) {
             minOrder = itemData.sku.def.quantityModule.minOrder.quantityFormatted;
          } else if (itemData.sku?.def?.quantityModule?.minOrder?.quantity) {
             minOrder = itemData.sku.def.quantityModule.minOrder.quantity;
          }

          let price = itemData.price || itemData.salePrice || itemData.sale_price || itemData.targetSalePrice || itemData.app_sale_price || itemData.price_with_currency || itemData.priceRange || itemData.minPrice || itemData.originalPrice || itemData.amount || "";
          if (itemData.sku?.def?.priceModule?.priceFormatted) {
             price = itemData.sku.def.priceModule.priceFormatted;
          } else if (itemData.sku?.def?.priceModule?.price) {
             price = "$" + itemData.sku.def.priceModule.price;
          }

          return {
            title: itemData.title || itemData.name || itemData.itemTitle || itemData.subject || itemData.productTitle || itemData.product_title || itemData.productName || "",
            price: price,
            minOrder,
            image: itemData.image || itemData.imageUrl || itemData.image_url || itemData.pic_url || itemData.picUrl || itemData.imgUrl || itemData.img_url || itemData.productImage || itemData.thumbnail || itemData.main_image || itemData.product_main_image_url
          }
        });
        console.log("Parsed Alibaba:", JSON.stringify(parsed, null, 2));

    } catch (e) {
        console.error(e);
    }
})();
