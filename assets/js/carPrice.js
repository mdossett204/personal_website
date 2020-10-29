$(function() {
	var numCols = ["mileage", "mpg", "engineSize"];
	
	function createTable(result){
		let $table = $("<table/>");
		$.each(result, function(colName, colValues){
			createRow($table, colName, colValues);	
		});
		$.each(numCols, function(_, colName){
			createRow($table, colName);
		});
		let $button = $("<button/>").text("Get Price")
									.attr({"id":"getPrice", "style":"margin-top:10px;"});
		let $col = $("<td/>").attr({"colspan": 2, "style": "text-align:center;"});
		let $row = $("<tr/>").append($col);
		$col.append($button);
		$table.append($row);
		return $table;
	}
	
	function createRow(table, colName, colValues){
		let $row = $("<tr/>");
		createLabelCol($row, colName);
		if (typeof colValues === "undefined"){
			createNumColInput($row, colName);
		} else {
			createOptionCol($row, colName, colValues);
		}
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
	
	function createNumColInput(row,colName){
		let $col = $("<td/>");
		let $input = $("<input/>").attr({type:"text", "name":colName, "id":colName});
		$col.append($input);
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
	
	function validateInput(name, input){
		if (numCols.includes(name)){
			if (input.match(/^\d+\.?\d*$/)){
				return parseFloat(input);
			} 
			alert("The input for "+ name.toUpperCase() + " needs to be a number.");
			return false;
		}
		return input;
	};
	
	$("#carData").on("click", "#getPrice", function(){
		var data = {};
		var success = true;
		$("#carData input").each(function(){
			let name = $(this).attr("name");
			let value = validateInput(name, $(this).val());
			if (value === false){
				success = false;
				return false;
			}
			data[name] = value;
		});
		if (!success){
			return;
		}
	});
});
