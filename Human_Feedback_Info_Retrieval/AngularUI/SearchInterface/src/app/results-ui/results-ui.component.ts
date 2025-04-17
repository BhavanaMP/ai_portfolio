import { HttpErrorResponse } from '@angular/common/http';
import { Component, Input, OnInit } from '@angular/core';
import {ActivatedRoute, Router, RouterModule} from '@angular/router';
import { from } from 'rxjs';
import { ReRank } from '../model/rerank.model';
import { Result } from '../model/results.model';
import { SearchUIComponent } from '../search-ui/search-ui.component';
import { ServletService } from '../Service/servlet.service';
import { DomSanitizer} from '@angular/platform-browser';
import { NgxSpinnerService } from "ngx-spinner";

@Component({
  selector: 'app-results-ui',
  templateUrl: './results-ui.component.html',
  styleUrls: ['./results-ui.component.css']
})
export class ResultsUiComponent implements OnInit {
  reRankobj: ReRank[] = [];
  results: Result[] = []; //array of objects of type Result
  public query: string = "";
  
  constructor(private service : ServletService,private spinner: NgxSpinnerService,private route: ActivatedRoute,
    private router: Router,private sanitizer:DomSanitizer) { }

  ngOnInit(): void {
    this.spinner.show();
 
    setTimeout(() => {
      /** spinner ends after 5 seconds */
      this.spinner.hide();
    }, 5000);
    this.query=this.service.getQuery();
    this.safePath(); 
    this.results=[];
    this.results=this.service.results;
    console.log(this.results)

  }
  onRateChange(event: number,title:any) {
    
    console.log("Rating changed", event);
    
    this.results.forEach(element => {
      console.log(element.title+"   "+title);
      if (element.title == title) {
        element.ratings=event;
      }
      
    });
    
    //this.results[rank-1].ratings=event;
    
  }
  logout(){
    console.log("Logging out,sending reset command,navigating to home page")

    this.service.reset("reset").subscribe(
    (data) => {});

    this.router.navigate([ '/home' ]);
   
  }
  safePath(){
    this.service.results.forEach(element => {
     // URL:string = ngx-print(element.imagepath);
     element.imagepath= this.sanitizer.bypassSecurityTrustUrl(element.imagepath);
     element.filepath= this.sanitizer.bypassSecurityTrustUrl(element.filepath);
      console.log(element.imagepath);
      
    });
   // return this.results;  
  }
  
  searchService() {
   // this.results =[];
    console.log("search in results page")
    if (this.query=="") {
      alert("Please enter query to proceed further")
      this.router.navigate(['/search']);
    } else {
      console.log("ïnside else")
      this.service.sendQuery(this.query)
      .subscribe(
          (data) => {
            this.results =[];
            console.log('Form submitted successfully'); 
            console.log(data); //change it to results variable of service class
            this.results=data;
            console.log(this.results.length);
            if (this.results.length==0) {
              alert("No Data found for given query.Please searchanother query")
              this.router.navigate(['/search']);
            } else {
              this.results =[];
              this.results=data;
              console.log("service call selse"+this.results)
              
              this.service.results=this.results;
              this.service.getobj(this.results);
                 this.router.navigate([ '/results' ])
            }
           
          },
          (error: HttpErrorResponse) => {
            console.log("inside error")
              console.log(error);
             // this.router.navigate([ '/results' ])
          }
      );  
      
    }
    
  }
  
reRank(){
  this.reRankobj = [];
  this.results.forEach(element => {
    console.log(element.ratings);
    
  });
  this.results.forEach(element => {
    this.reRankobj.push(
      {
        "query":this.query,
       "ratings":element.ratings,
       "title":element.title //title
      });
    });
  console.log("Rerank object"+this.reRankobj);
  
  this.service.reRank(this.reRankobj)
  .subscribe(
    (data) => {
      console.log("data retrieved after rerank"+data);
      this.results = [];
      this.results=data;
      this.results.forEach(element => {
        console.log(element)
      });
      this.router.routeReuseStrategy.shouldReuseRoute = () => false;
      // this.router.onSameUrlNavigation = 'reload';
      // this.router.navigate(['/results'], { relativeTo: this.route });
     // this.ngOnInit();
    });

}


}

