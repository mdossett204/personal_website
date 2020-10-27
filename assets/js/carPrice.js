$(function() {
	
	function createTable(result){
		let $table = $("<table/>");
		$.each(result, function(colName, colValues){
			createRow($table, colName, colValues);	
		});
		return $table;
	}
	
	function createRow(table, colName, colValues){
		let $row = $("<tr/>");
		createLabelCol($row, colName);
		createOptionCol($row, colName, colValues);
		table.append($row);
	}
	
	function createLabelCol(row, colName){
		let $col = $("<td/>").text(colName.toUpperCase());
		row.append($col);
	}
	
	function createOptionCol(row, colName, colValues){
		let $col = $("<td/>");
		let $input = $("<input/>").attr({"list": colName+"s", "name":colName, "id":colName});
		$col.append($input);
		let $dataList = $("<datalist/>").attr("id", colName+"s");
		createOptions($dataList, colValues);
		$col.append($dataList);
		row.append($col);
	}
	
	function createOptions(dataList,options){
		options.forEach(function(option){
			let	$option = $("<option/>").attr({"value":option});
			dataList.append($option);
		});
	}
	
	$.ajax({
  		dataType: "json",
  		url: "/assets/js/feature_options.json",
  		success: function(result){
  			$("#carData").append(createTable(result));
  			alert("Success!");
  		}
	});
	
});
